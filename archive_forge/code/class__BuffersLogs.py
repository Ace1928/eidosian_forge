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
@attr.s(auto_attribs=True)
class _BuffersLogs:
    """
    A context manager that buffers any log events until after its
    block exits.

    @ivar _namespace: The namespace of the buffered events.
    @type _namespace: L{str}.

    @ivar _observer: The observer to which buffered log events will be
        written
    @type _observer: L{twisted.logger.ILogObserver}.
    """
    _namespace: str
    _observer: ILogObserver
    _logs: List[LogEvent] = attr.ib(default=attr.Factory(list))

    def __enter__(self):
        """
        Enter a log buffering context.

        @return: A logger that buffers log events.
        @rtype: L{Logger}.
        """
        return Logger(namespace=self._namespace, observer=self._logs.append)

    def __exit__(self, excValue, excType, traceback):
        """
        Exit a log buffering context and log all buffered events to
        the provided observer.

        @param excType: See L{object.__exit__}
        @param excValue: See L{object.__exit__}
        @param traceback: See L{object.__exit__}
        """
        for event in self._logs:
            self._observer(event)