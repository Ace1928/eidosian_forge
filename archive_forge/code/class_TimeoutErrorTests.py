from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
class TimeoutErrorTests(unittest.TestCase, ImmediateFailureMixin):
    """
    L{twisted.internet.defer} timeout code.
    """

    def test_deprecatedTimeout(self) -> None:
        """
        L{twisted.internet.defer.timeout} is deprecated.
        """
        deferred: Deferred[object] = Deferred()
        defer.timeout(deferred)
        self.assertFailure(deferred, defer.TimeoutError)
        warningsShown = self.flushWarnings([self.test_deprecatedTimeout])
        self.assertEqual(len(warningsShown), 1)
        self.assertIs(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], 'twisted.internet.defer.timeout was deprecated in Twisted 17.1.0; please use twisted.internet.defer.Deferred.addTimeout instead')