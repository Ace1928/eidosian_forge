from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _stopSimulating(self) -> None:
    """
        If we have a CFRunLoopTimer registered with the CFRunLoop, invalidate
        it and set it to None.
        """
    if self._currentSimulator is None:
        return
    CFRunLoopTimerInvalidate(self._currentSimulator)
    self._currentSimulator = None