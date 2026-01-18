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
@implementer(_IWaker)
class _UnixWaker(_FDWaker):
    """
    This class provides a simple interface to wake up the event loop.

    This is used by threads or signals to wake up the event loop.
    """

    def wakeUp(self):
        """Write one byte to the pipe, and flush it."""
        if self.o is not None:
            try:
                util.untilConcludes(os.write, self.o, b'x')
            except OSError as e:
                if e.errno != errno.EAGAIN:
                    raise