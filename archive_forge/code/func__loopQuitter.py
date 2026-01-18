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
def _loopQuitter(idleAdd: Callable[[Callable[[], None]], None], loopQuit: Callable[[], None]) -> Callable[[], None]:
    """
    Combine the C{glib.idle_add} and C{glib.MainLoop.quit} functions into a
    function suitable for crashing the reactor.
    """
    return lambda: idleAdd(loopQuit)