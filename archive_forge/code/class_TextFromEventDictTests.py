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
class TextFromEventDictTests(unittest.SynchronousTestCase):
    """
    Tests for L{textFromEventDict}.
    """

    def test_message(self) -> None:
        """
        The C{"message"} value, when specified, is concatenated to generate the
        message.
        """
        eventDict = dict(message=('a', 'b', 'c'))
        text = log.textFromEventDict(eventDict)
        self.assertEqual(text, 'a b c')

    def test_format(self) -> None:
        """
        The C{"format"} value, when specified, is used to format the message.
        """
        eventDict = dict(message=(), isError=0, format='Hello, %(foo)s!', foo='dude')
        text = log.textFromEventDict(eventDict)
        self.assertEqual(text, 'Hello, dude!')

    def test_noMessageNoFormat(self) -> None:
        """
        If C{"format"} is unspecified and C{"message"} is empty, return
        L{None}.
        """
        eventDict = dict(message=(), isError=0)
        text = log.textFromEventDict(eventDict)
        self.assertIsNone(text)

    def test_whySpecified(self) -> None:
        """
        The C{"why"} value, when specified, is first part of message.
        """
        try:
            raise RuntimeError()
        except BaseException:
            eventDict = dict(message=(), isError=1, failure=failure.Failure(), why='foo')
            text = log.textFromEventDict(eventDict)
            assert text is not None
            self.assertTrue(text.startswith('foo\n'))

    def test_whyDefault(self) -> None:
        """
        The C{"why"} value, when unspecified, defaults to C{"Unhandled Error"}.
        """
        try:
            raise RuntimeError()
        except BaseException:
            eventDict = dict(message=(), isError=1, failure=failure.Failure())
            text = log.textFromEventDict(eventDict)
            assert text is not None
            self.assertTrue(text.startswith('Unhandled Error\n'))

    def test_noTracebackForYou(self) -> None:
        """
        If unable to obtain a traceback due to an exception, catch it and note
        the error.
        """
        eventDict = dict(message=(), isError=1, failure=object())
        text = log.textFromEventDict(eventDict)
        self.assertIn('\n(unable to obtain traceback)', text)