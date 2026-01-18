from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
class PythonLoggingObserverTests(unittest.SynchronousTestCase):
    """
    Test the bridge with python logging module.
    """

    def setUp(self) -> None:
        rootLogger = logging.getLogger('')
        originalLevel = rootLogger.getEffectiveLevel()
        rootLogger.setLevel(logging.DEBUG)

        @self.addCleanup
        def restoreLevel() -> None:
            rootLogger.setLevel(originalLevel)
        self.hdlr, self.out = handlerAndBytesIO()
        rootLogger.addHandler(self.hdlr)

        @self.addCleanup
        def removeLogger() -> None:
            rootLogger.removeHandler(self.hdlr)
            self.hdlr.close()
        self.lp = log.LogPublisher()
        self.obs = log.PythonLoggingObserver()
        self.lp.addObserver(self.obs.emit)

    def test_singleString(self) -> None:
        """
        Test simple output, and default log level.
        """
        self.lp.msg('Hello, world.')
        self.assertIn(b'Hello, world.', self.out.getvalue())
        self.assertIn(b'INFO', self.out.getvalue())

    def test_errorString(self) -> None:
        """
        Test error output.
        """
        f = failure.Failure(ValueError('That is bad.'))
        self.lp.msg(failure=f, isError=True)
        prefix = b'CRITICAL:'
        output = self.out.getvalue()
        self.assertTrue(output.startswith(prefix), f'Does not start with {prefix!r}: {output!r}')

    def test_formatString(self) -> None:
        """
        Test logging with a format.
        """
        self.lp.msg(format='%(bar)s oo %(foo)s', bar='Hello', foo='world')
        self.assertIn(b'Hello oo world', self.out.getvalue())

    def test_customLevel(self) -> None:
        """
        Test the logLevel keyword for customizing level used.
        """
        self.lp.msg('Spam egg.', logLevel=logging.ERROR)
        self.assertIn(b'Spam egg.', self.out.getvalue())
        self.assertIn(b'ERROR', self.out.getvalue())
        self.out.seek(0, 0)
        self.out.truncate()
        self.lp.msg('Foo bar.', logLevel=logging.WARNING)
        self.assertIn(b'Foo bar.', self.out.getvalue())
        self.assertIn(b'WARNING', self.out.getvalue())

    def test_strangeEventDict(self) -> None:
        """
        Verify that an event dictionary which is not an error and has an empty
        message isn't recorded.
        """
        self.lp.msg(message='', isError=False)
        self.assertEqual(self.out.getvalue(), b'')