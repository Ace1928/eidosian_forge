import gireactor or gtk3reactor for GObject Introspection based applications,
import sys
from typing import Any, Callable, Dict, Set
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor, IWriteDescriptor
from twisted.python import log
from twisted.python.monkey import MonkeyPatcher
from ._signals import _IWaker, _UnixWaker
def _ioEventCallback(self, source, condition):
    """
        Called by event loop when an I/O event occurs.
        """
    log.callWithLogger(source, self._doReadOrWrite, source, source, condition)
    return True