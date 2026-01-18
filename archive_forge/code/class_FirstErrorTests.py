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
class FirstErrorTests(unittest.SynchronousTestCase):
    """
    Tests for L{FirstError}.
    """

    def test_repr(self) -> None:
        """
        The repr of a L{FirstError} instance includes the repr of the value of
        the sub-failure and the index which corresponds to the L{FirstError}.
        """
        exc = ValueError('some text')
        try:
            raise exc
        except BaseException:
            f = Failure()
        error = defer.FirstError(f, 3)
        self.assertEqual(repr(error), f'FirstError[#3, {repr(exc)}]')

    def test_str(self) -> None:
        """
        The str of a L{FirstError} instance includes the str of the
        sub-failure and the index which corresponds to the L{FirstError}.
        """
        exc = ValueError('some text')
        try:
            raise exc
        except BaseException:
            f = Failure()
        error = defer.FirstError(f, 5)
        self.assertEqual(str(error), f'FirstError[#5, {str(f)}]')

    def test_comparison(self) -> None:
        """
        L{FirstError} instances compare equal to each other if and only if
        their failure and index compare equal.  L{FirstError} instances do not
        compare equal to instances of other types.
        """
        try:
            1 // 0
        except BaseException:
            firstFailure = Failure()
        one = defer.FirstError(firstFailure, 13)
        anotherOne = defer.FirstError(firstFailure, 13)
        try:
            raise ValueError('bar')
        except BaseException:
            secondFailure = Failure()
        another = defer.FirstError(secondFailure, 9)
        self.assertTrue(one == anotherOne)
        self.assertFalse(one == another)
        self.assertTrue(one != another)
        self.assertFalse(one != anotherOne)
        self.assertFalse(one == 10)