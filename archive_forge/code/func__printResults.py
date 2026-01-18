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
def _printResults(self, flavor, errors, formatter):
    """
        Print a group of errors to the stream.

        @param flavor: A string indicating the kind of error (e.g. 'TODO').
        @param errors: A list of errors, often L{failure.Failure}s, but
            sometimes 'todo' errors.
        @param formatter: A callable that knows how to format the errors.
        """
    for reason, cases in self._groupResults(errors, formatter):
        self._writeln(self._doubleSeparator)
        self._writeln(flavor)
        self._write(reason)
        self._writeln('')
        for case in cases:
            self._writeln(case.id())