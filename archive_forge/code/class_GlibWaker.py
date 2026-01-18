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
class GlibWaker(_UnixWaker):
    """
    Run scheduled events after waking up.
    """

    def __init__(self, reactor):
        super().__init__()
        self.reactor = reactor

    def doRead(self) -> None:
        super().doRead()
        self.reactor._simulate()