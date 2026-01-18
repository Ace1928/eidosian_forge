from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
class HTTPFactory(protocol.ServerFactory):
    """
    Factory for HTTP server.

    @ivar _logDateTime: A cached datetime string for log messages, updated by
        C{_logDateTimeCall}.
    @type _logDateTime: C{str}

    @ivar _logDateTimeCall: A delayed call for the next update to the cached
        log datetime string.
    @type _logDateTimeCall: L{IDelayedCall} provided

    @ivar _logFormatter: See the C{logFormatter} parameter to L{__init__}

    @ivar _nativeize: A flag that indicates whether the log file being written
        to wants native strings (C{True}) or bytes (C{False}).  This is only to
        support writing to L{twisted.python.log} which, unfortunately, works
        with native strings.

    @ivar reactor: An L{IReactorTime} provider used to manage connection
        timeouts and compute logging timestamps.
    """
    protocol = _genericHTTPChannelProtocolFactory
    logPath = None
    timeOut = _REQUEST_TIMEOUT

    def __init__(self, logPath=None, timeout=_REQUEST_TIMEOUT, logFormatter=None, reactor=None):
        """
        @param logPath: File path to which access log messages will be written
            or C{None} to disable logging.
        @type logPath: L{str} or L{bytes}

        @param timeout: The initial value of L{timeOut}, which defines the idle
            connection timeout in seconds, or C{None} to disable the idle
            timeout.
        @type timeout: L{float}

        @param logFormatter: An object to format requests into log lines for
            the access log.  L{combinedLogFormatter} when C{None} is passed.
        @type logFormatter: L{IAccessLogFormatter} provider

        @param reactor: An L{IReactorTime} provider used to manage connection
            timeouts and compute logging timestamps. Defaults to the global
            reactor.
        """
        if not reactor:
            from twisted.internet import reactor
        self.reactor = reactor
        if logPath is not None:
            logPath = os.path.abspath(logPath)
        self.logPath = logPath
        self.timeOut = timeout
        if logFormatter is None:
            logFormatter = combinedLogFormatter
        self._logFormatter = logFormatter
        self._logDateTime = None
        self._logDateTimeCall = None

    def _updateLogDateTime(self):
        """
        Update log datetime periodically, so we aren't always recalculating it.
        """
        self._logDateTime = datetimeToLogString(self.reactor.seconds())
        self._logDateTimeCall = self.reactor.callLater(1, self._updateLogDateTime)

    def buildProtocol(self, addr):
        p = protocol.ServerFactory.buildProtocol(self, addr)
        p.callLater = self.reactor.callLater
        p.timeOut = self.timeOut
        return p

    def startFactory(self):
        """
        Set up request logging if necessary.
        """
        if self._logDateTimeCall is None:
            self._updateLogDateTime()
        if self.logPath:
            self.logFile = self._openLogFile(self.logPath)
        else:
            self.logFile = log.logfile

    def stopFactory(self):
        if hasattr(self, 'logFile'):
            if self.logFile != log.logfile:
                self.logFile.close()
            del self.logFile
        if self._logDateTimeCall is not None and self._logDateTimeCall.active():
            self._logDateTimeCall.cancel()
            self._logDateTimeCall = None

    def _openLogFile(self, path):
        """
        Override in subclasses, e.g. to use L{twisted.python.logfile}.
        """
        f = open(path, 'ab', 1)
        return f

    def log(self, request):
        """
        Write a line representing C{request} to the access log file.

        @param request: The request object about which to log.
        @type request: L{Request}
        """
        try:
            logFile = self.logFile
        except AttributeError:
            pass
        else:
            line = self._logFormatter(self._logDateTime, request) + '\n'
            logFile.write(line.encode('utf8'))