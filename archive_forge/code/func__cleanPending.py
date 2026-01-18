from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _cleanPending(self):
    """
        Cancel all pending calls and return their string representations.
        """
    reactor = self._getReactor()
    reactor.iterate(0)
    reactor.iterate(0)
    delayedCallStrings = []
    for p in reactor.getDelayedCalls():
        if p.active():
            delayedString = str(p)
            p.cancel()
        else:
            print('WEIRDNESS! pending timed call not active!')
        delayedCallStrings.append(delayedString)
    return delayedCallStrings