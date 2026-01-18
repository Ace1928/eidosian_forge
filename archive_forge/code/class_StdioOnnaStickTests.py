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
class StdioOnnaStickTests(unittest.SynchronousTestCase):
    """
    StdioOnnaStick should act like the normal sys.stdout object.
    """

    def setUp(self) -> None:
        self.resultLogs: list[log.EventDict] = []
        log.addObserver(self.resultLogs.append)

    def tearDown(self) -> None:
        log.removeObserver(self.resultLogs.append)

    def getLogMessages(self) -> list[str]:
        return [''.join(d['message']) for d in self.resultLogs]

    def test_write(self) -> None:
        """
        Writing to a StdioOnnaStick instance results in Twisted log messages.

        Log messages are generated every time a '\\n' is encountered.
        """
        stdio = log.StdioOnnaStick()
        stdio.write('Hello there\nThis is a test')
        self.assertEqual(self.getLogMessages(), ['Hello there'])
        stdio.write('!\n')
        self.assertEqual(self.getLogMessages(), ['Hello there', 'This is a test!'])

    def test_metadata(self) -> None:
        """
        The log messages written by StdioOnnaStick have printed=1 keyword, and
        by default are not errors.
        """
        stdio = log.StdioOnnaStick()
        stdio.write('hello\n')
        self.assertFalse(self.resultLogs[0]['isError'])
        self.assertTrue(self.resultLogs[0]['printed'])

    def test_writeLines(self) -> None:
        """
        Writing lines to a StdioOnnaStick results in Twisted log messages.
        """
        stdio = log.StdioOnnaStick()
        stdio.writelines(['log 1', 'log 2'])
        self.assertEqual(self.getLogMessages(), ['log 1', 'log 2'])

    def test_print(self) -> None:
        """
        When StdioOnnaStick is set as sys.stdout, prints become log messages.
        """
        oldStdout = sys.stdout
        sys.stdout = log.StdioOnnaStick()
        self.addCleanup(setattr, sys, 'stdout', oldStdout)
        print('This', end=' ')
        print('is a test')
        self.assertEqual(self.getLogMessages(), ['This is a test'])

    def test_error(self) -> None:
        """
        StdioOnnaStick created with isError=True log messages as errors.
        """
        stdio = log.StdioOnnaStick(isError=True)
        stdio.write('log 1\n')
        self.assertTrue(self.resultLogs[0]['isError'])

    def test_unicode(self) -> None:
        """
        StdioOnnaStick converts unicode prints to byte strings on Python 2, in
        order to be compatible with the normal stdout/stderr objects.

        On Python 3, the prints are left unmodified.
        """
        unicodeString = 'Hello, ½ world.'
        stdio = log.StdioOnnaStick(encoding='utf-8')
        self.assertEqual(stdio.encoding, 'utf-8')
        stdio.write(unicodeString + '\n')
        stdio.writelines(['Also, ' + unicodeString])
        oldStdout = sys.stdout
        sys.stdout = stdio
        self.addCleanup(setattr, sys, 'stdout', oldStdout)
        print(unicodeString)
        self.assertEqual(self.getLogMessages(), [unicodeString, 'Also, ' + unicodeString, unicodeString])