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
class TimingTextReporter(VerboseTextReporter):
    """
    Prints out each test as it is running, followed by the time taken for each
    test to run.
    """

    def stopTest(self, method):
        """
        Mark the test as stopped, and write the time it took to run the test
        to the stream.
        """
        super().stopTest(method)
        self._write('(%.03f secs)\n' % self._lastTime)