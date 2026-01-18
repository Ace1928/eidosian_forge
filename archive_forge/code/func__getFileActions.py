from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _getFileActions(fdState: List[Tuple[int, bool]], childToParentFD: Dict[int, int], doClose: int, doDup2: int) -> List[Tuple[int, ...]]:
    """
    Get the C{file_actions} parameter for C{posix_spawn} based on the
    parameters describing the current process state.

    @param fdState: A list of 2-tuples of (file descriptor, close-on-exec
        flag).

    @param doClose: the integer to use for the 'close' instruction

    @param doDup2: the integer to use for the 'dup2' instruction
    """
    fdStateDict = dict(fdState)
    parentToChildren: Dict[int, List[int]] = defaultdict(list)
    for inChild, inParent in childToParentFD.items():
        parentToChildren[inParent].append(inChild)
    allocated = set(fdStateDict)
    allocated |= set(childToParentFD.values())
    allocated |= set(childToParentFD.keys())
    nextFD = 0

    def allocateFD() -> int:
        nonlocal nextFD
        while nextFD in allocated:
            nextFD += 1
        allocated.add(nextFD)
        return nextFD
    result: List[Tuple[int, ...]] = []
    relocations = {}
    for inChild, inParent in sorted(childToParentFD.items()):
        parentToChildren[inParent].remove(inChild)
        if parentToChildren[inChild]:
            new = relocations[inChild] = allocateFD()
            result.append((doDup2, inChild, new))
        if inParent in relocations:
            result.append((doDup2, relocations[inParent], inChild))
            if not parentToChildren[inParent]:
                result.append((doClose, relocations[inParent]))
        elif inParent == inChild:
            if fdStateDict[inParent]:
                tempFD = allocateFD()
                result.extend([(doDup2, inParent, tempFD), (doDup2, tempFD, inChild), (doClose, tempFD)])
        else:
            result.append((doDup2, inParent, inChild))
    for eachFD, uninheritable in fdStateDict.items():
        if eachFD not in childToParentFD and (not uninheritable):
            result.append((doClose, eachFD))
    return result