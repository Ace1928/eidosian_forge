from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _cleanReactor(self):
    """
        Remove all selectables from the reactor, kill any of them that were
        processes, and return their string representation.
        """
    reactor = self._getReactor()
    selectableStrings = []
    for sel in reactor.removeAll():
        if interfaces.IProcessTransport.providedBy(sel):
            sel.signalProcess('KILL')
        selectableStrings.append(repr(sel))
    return selectableStrings