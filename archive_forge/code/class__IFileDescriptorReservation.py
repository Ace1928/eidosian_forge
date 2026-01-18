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
class _IFileDescriptorReservation(Interface):
    """
    An open file that represents an emergency reservation in the
    process' file descriptor table.  If L{Port} encounters C{EMFILE}
    on C{accept(2)}, it can close this file descriptor, retry the
    C{accept} so that the incoming connection occupies this file
    descriptor's space, and then close that connection and reopen this
    one.

    Calling L{_IFileDescriptorReservation.reserve} attempts to open
    the reserve file descriptor if it is not already open.
    L{_IFileDescriptorReservation.available} returns L{True} if the
    underlying file is open and its descriptor claimed.

    L{_IFileDescriptorReservation} instances are context managers;
    entering them releases the underlying file descriptor, while
    exiting them attempts to reacquire it.  The block can take
    advantage of the free slot in the process' file descriptor table
    accept and close a client connection.

    Because another thread might open a file descriptor between the
    time the context manager is entered and the time C{accept} is
    called, opening the reserve descriptor is best-effort only.
    """

    def available():
        """
        Is the reservation available?

        @return: L{True} if the reserved file descriptor is open and
            can thus be closed to allow a new file to be opened in its
            place; L{False} if it is not open.
        """

    def reserve():
        """
        Attempt to open the reserved file descriptor; if this fails
        because of C{EMFILE}, internal state is reset so that another
        reservation attempt can be made.

        @raises Exception: Any exception except an L{OSError} whose
            errno is L{EMFILE}.
        """

    def __enter__():
        """
        Release the underlying file descriptor so that code within the
        context manager can open a new file.
        """

    def __exit__(excType, excValue, traceback):
        """
        Attempt to re-open the reserved file descriptor.  See
        L{reserve} for caveats.

        @param excType: See L{object.__exit__}
        @param excValue: See L{object.__exit__}
        @param traceback: See L{object.__exit__}
        """