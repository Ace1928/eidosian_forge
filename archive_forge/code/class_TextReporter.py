from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
class TextReporter(Reporter):
    """
    Simple reporter that prints a single character for each test as it runs,
    along with the standard Trial summary text.
    """

    def addSuccess(self, test):
        super().addSuccess(test)
        self._write('.')

    def addError(self, *args):
        super().addError(*args)
        self._write('E')

    def addFailure(self, *args):
        super().addFailure(*args)
        self._write('F')

    def addSkip(self, *args):
        super().addSkip(*args)
        self._write('S')

    def addExpectedFailure(self, *args):
        super().addExpectedFailure(*args)
        self._write('T')

    def addUnexpectedSuccess(self, *args):
        super().addUnexpectedSuccess(*args)
        self._write('!')