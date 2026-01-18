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
@frozen
class _WithSignalHandling:
    """
    A reactor core helper that can manage signals: it installs signal handlers
    at start time.
    """
    _sigInt: SignalHandler
    _sigBreak: SignalHandler
    _sigTerm: SignalHandler

    def install(self) -> None:
        """
        Install the signal handlers for the Twisted event loop.
        """
        if signal.getsignal(signal.SIGINT) == signal.default_int_handler:
            signal.signal(signal.SIGINT, self._sigInt)
        signal.signal(signal.SIGTERM, self._sigTerm)
        SIGBREAK = getattr(signal, 'SIGBREAK', None)
        if SIGBREAK is not None:
            signal.signal(SIGBREAK, self._sigBreak)

    def uninstall(self) -> None:
        """
        At the moment, do nothing (for historical reasons).
        """