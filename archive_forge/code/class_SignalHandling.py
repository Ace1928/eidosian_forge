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
class SignalHandling(Protocol):
    """
    The L{SignalHandling} protocol enables customizable signal-handling
    behaviors for reactors.

    A value that conforms to L{SignalHandling} has install and uninstall hooks
    that are called by a reactor at the correct times to have the (typically)
    process-global effects necessary for dealing with signals.
    """

    def install(self) -> None:
        """
        Install the signal handlers.
        """

    def uninstall(self) -> None:
        """
        Restore signal handlers to their original state.
        """