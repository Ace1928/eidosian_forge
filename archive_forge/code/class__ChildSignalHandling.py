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
class _ChildSignalHandling:
    """
    Signal handling behavior which supports I{SIGCHLD} for notification about
    changes to child process state.

    @ivar _childWaker: L{None} or a reference to the L{_SIGCHLDWaker} which is
        used to properly notice child process termination.  This is L{None}
        when this handling behavior is not installed and non-C{None}
        otherwise.  This is mostly an unfortunate implementation detail due to
        L{_SIGCHLDWaker} allocating file descriptors as a side-effect of its
        initializer.
    """
    _addInternalReader: Callable[[IReadDescriptor], object]
    _removeInternalReader: Callable[[IReadDescriptor], object]
    _childWaker: Optional[_SIGCHLDWaker] = None

    def install(self) -> None:
        """
        Extend the basic signal handling logic to also support handling
        SIGCHLD to know when to try to reap child processes.
        """
        if self._childWaker is None:
            self._childWaker = _SIGCHLDWaker()
            self._addInternalReader(self._childWaker)
        self._childWaker.install()
        process.reapAllProcesses()

    def uninstall(self) -> None:
        """
        If a child waker was created and installed, uninstall it now.

        Since this disables reactor functionality and is only called when the
        reactor is stopping, it doesn't provide any directly useful
        functionality, but the cleanup of reactor-related process-global state
        that it does helps in unit tests involving multiple reactors and is
        generally just a nice thing.
        """
        assert self._childWaker is not None
        self._removeInternalReader(self._childWaker)
        self._childWaker.uninstall()
        self._childWaker.connectionLost(failure.Failure(Exception('uninstalled')))
        self._childWaker = None