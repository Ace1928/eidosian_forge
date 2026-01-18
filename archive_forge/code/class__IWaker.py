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
class _IWaker(Interface):
    """
    Interface to wake up the event loop based on the self-pipe trick.

    The U{I{self-pipe trick}<http://cr.yp.to/docs/selfpipe.html>}, used to wake
    up the main loop from another thread or a signal handler.
    This is why we have wakeUp together with doRead

    This is used by threads or signals to wake up the event loop.
    """
    disconnected = Attribute('')

    def wakeUp() -> None:
        """
        Called when the event should be wake up.
        """

    def doRead() -> None:
        """
        Read some data from my connection and discard it.
        """

    def connectionLost(reason: failure.Failure) -> None:
        """
        Called when connection was closed and the pipes.
        """