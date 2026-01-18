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
def _finishInit(self, whenDone, skt, error, reactor):
    """
        Called by subclasses to continue to the stage of initialization where
        the socket connect attempt is made.

        @param whenDone: A 0-argument callable to invoke once the connection is
            set up.  This is L{None} if the connection could not be prepared
            due to a previous error.

        @param skt: The socket object to use to perform the connection.
        @type skt: C{socket._socketobject}

        @param error: The error to fail the connection with.

        @param reactor: The reactor to use for this client.
        @type reactor: L{twisted.internet.interfaces.IReactorTime}
        """
    if whenDone:
        self._commonConnection.__init__(self, skt, None, reactor)
        reactor.callLater(0, whenDone)
    else:
        reactor.callLater(0, self.failIfNotConnected, error)