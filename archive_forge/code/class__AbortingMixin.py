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
class _AbortingMixin:
    """
    Common implementation of C{abortConnection}.

    @ivar _aborting: Set to C{True} when C{abortConnection} is called.
    @type _aborting: C{bool}
    """
    _aborting = False

    def abortConnection(self):
        """
        Aborts the connection immediately, dropping any buffered data.

        @since: 11.1
        """
        if self.disconnected or self._aborting:
            return
        self._aborting = True
        self.stopReading()
        self.stopWriting()
        self.doRead = lambda *args, **kwargs: None
        self.doWrite = lambda *args, **kwargs: None
        self.reactor.callLater(0, self.connectionLost, failure.Failure(error.ConnectionAborted()))