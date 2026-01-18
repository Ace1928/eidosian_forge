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
class LogPublisherTestCaseMixin:

    def setUp(self) -> None:
        """
        Add a log observer which records log events in C{self.out}.  Also,
        make sure the default string encoding is ASCII so that
        L{testSingleUnicode} can test the behavior of logging unencodable
        unicode messages.
        """
        self.out = FakeFile()
        self.lp = log.LogPublisher()
        self.flo = log.FileLogObserver(self.out)
        self.lp.addObserver(self.flo.emit)

    def tearDown(self: _LogPublisherTestCaseMixinBase) -> None:
        """
        Verify that everything written to the fake file C{self.out} was a
        C{str}.  Also, restore the default string encoding to its previous
        setting, if it was modified by L{setUp}.
        """
        for chunk in self.out:
            self.assertIsInstance(chunk, str, f'{chunk!r} was not a string')