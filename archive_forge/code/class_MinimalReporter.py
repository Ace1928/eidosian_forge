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
class MinimalReporter(Reporter):
    """
    A minimalist reporter that prints only a summary of the test result, in
    the form of (timeTaken, #tests, #tests, #errors, #failures, #skips).
    """

    def _printErrors(self):
        """
        Don't print a detailed summary of errors. We only care about the
        counts.
        """

    def _printSummary(self):
        """
        Print out a one-line summary of the form:
        '%(runtime) %(number_of_tests) %(number_of_tests) %(num_errors)
        %(num_failures) %(num_skips)'
        """
        numTests = self.testsRun
        if self._startTime is not None:
            timing = self._getTime() - self._startTime
        else:
            timing = 0
        t = (timing, numTests, numTests, len(self.errors), len(self.failures), len(self.skips))
        self._writeln(' '.join(map(str, t)))