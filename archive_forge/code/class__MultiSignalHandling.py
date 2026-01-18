from __future__ import annotations
import contextlib
import errno
import os
import signal
import socket
from types import FrameType
from typing import Callable, Optional, Sequence
from zope.interface import Attribute, Interface, implementer
from attrs import define, frozen
from typing_extensions import Protocol, TypeAlias
from twisted.internet.interfaces import IReadDescriptor
from twisted.python import failure, log, util
from twisted.python.runtime import platformType
@define
class _MultiSignalHandling:
    """
    An implementation of L{SignalHandling} which propagates protocol
    method calls to a number of other implementations.

    This supports composition of multiple signal handling implementations into
    a single object so the reactor doesn't have to be concerned with how those
    implementations are factored.

    @ivar _signalHandlings: The other C{SignalHandling} implementations to
        which to propagate calls.

    @ivar _installed: If L{install} has been called but L{uninstall} has not.
        This is used to avoid double cleanup which otherwise results (at least
        during test suite runs) because twisted.internet.reactormixins doesn't
        keep track of whether a reactor has run or not but always invokes its
        cleanup logic.
    """
    _signalHandlings: Sequence[SignalHandling]
    _installed: bool = False

    def install(self) -> None:
        for d in self._signalHandlings:
            d.install()
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            for d in self._signalHandlings:
                d.uninstall()
            self._installed = False