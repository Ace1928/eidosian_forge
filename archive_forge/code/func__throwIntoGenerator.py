from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
def _throwIntoGenerator(self, f: failure.Failure, g: Generator[Any, Any, Any]) -> None:
    try:
        f.throwExceptionIntoGenerator(g)
    except StopIteration:
        pass
    else:
        self.fail('throwExceptionIntoGenerator should have raised StopIteration')