import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application
import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType
class ExpectLog(logging.Filter):
    """Context manager to capture and suppress expected log output.

    Useful to make tests of error conditions less noisy, while still
    leaving unexpected log entries visible.  *Not thread safe.*

    The attribute ``logged_stack`` is set to ``True`` if any exception
    stack trace was logged.

    Usage::

        with ExpectLog('tornado.application', "Uncaught exception"):
            error_response = self.fetch("/some_page")

    .. versionchanged:: 4.3
       Added the ``logged_stack`` attribute.
    """

    def __init__(self, logger: Union[logging.Logger, basestring_type], regex: str, required: bool=True, level: Optional[int]=None) -> None:
        """Constructs an ExpectLog context manager.

        :param logger: Logger object (or name of logger) to watch.  Pass an
            empty string to watch the root logger.
        :param regex: Regular expression to match.  Any log entries on the
            specified logger that match this regex will be suppressed.
        :param required: If true, an exception will be raised if the end of the
            ``with`` statement is reached without matching any log entries.
        :param level: A constant from the ``logging`` module indicating the
            expected log level. If this parameter is provided, only log messages
            at this level will be considered to match. Additionally, the
            supplied ``logger`` will have its level adjusted if necessary (for
            the duration of the ``ExpectLog`` to enable the expected message.

        .. versionchanged:: 6.1
           Added the ``level`` parameter.

        .. deprecated:: 6.3
           In Tornado 7.0, only ``WARNING`` and higher logging levels will be
           matched by default. To match ``INFO`` and lower levels, the ``level``
           argument must be used. This is changing to minimize differences
           between ``tornado.testing.main`` (which enables ``INFO`` logs by
           default) and most other test runners (including those in IDEs)
           which have ``INFO`` logs disabled by default.
        """
        if isinstance(logger, basestring_type):
            logger = logging.getLogger(logger)
        self.logger = logger
        self.regex = re.compile(regex)
        self.required = required
        self.matched = 0
        self.deprecated_level_matched = 0
        self.logged_stack = False
        self.level = level
        self.orig_level = None

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            self.logged_stack = True
        message = record.getMessage()
        if self.regex.match(message):
            if self.level is None and record.levelno < logging.WARNING:
                self.deprecated_level_matched += 1
            if self.level is not None and record.levelno != self.level:
                app_log.warning('Got expected log message %r at unexpected level (%s vs %s)' % (message, logging.getLevelName(self.level), record.levelname))
                return True
            self.matched += 1
            return False
        return True

    def __enter__(self) -> 'ExpectLog':
        if self.level is not None and self.level < self.logger.getEffectiveLevel():
            self.orig_level = self.logger.level
            self.logger.setLevel(self.level)
        self.logger.addFilter(self)
        return self

    def __exit__(self, typ: 'Optional[Type[BaseException]]', value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        if self.orig_level is not None:
            self.logger.setLevel(self.orig_level)
        self.logger.removeFilter(self)
        if not typ and self.required and (not self.matched):
            raise Exception('did not get expected log message')
        if not typ and self.required and (self.deprecated_level_matched >= self.matched):
            warnings.warn('ExpectLog matched at INFO or below without level argument', DeprecationWarning)