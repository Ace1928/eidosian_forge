from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
@implementer(_IFileDescriptorReservation)
class _NullFileDescriptorReservation:
    """
    A null implementation of L{_IFileDescriptorReservation}.
    """

    def available(self):
        """
        The reserved file is never available.  See
        L{_IFileDescriptorReservation.available}.

        @return: L{False}
        """
        return False

    def reserve(self):
        """
        Do nothing.  See L{_IFileDescriptorReservation.reserve}.
        """

    def __enter__(self):
        """
        Do nothing. See L{_IFileDescriptorReservation.__enter__}

        @return: L{False}
        """

    def __exit__(self, excType, excValue, traceback):
        """
        Do nothing.  See L{_IFileDescriptorReservation.__exit__}.

        @param excType: See L{object.__exit__}
        @param excValue: See L{object.__exit__}
        @param traceback: See L{object.__exit__}
        """