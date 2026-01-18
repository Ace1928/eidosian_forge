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
class EnsureDeferredTests(unittest.TestCase):
    """
    Tests for L{ensureDeferred}.
    """

    def test_passesThroughDeferreds(self) -> None:
        """
        L{ensureDeferred} will pass through a Deferred unchanged.
        """
        d1: Deferred[None] = Deferred()
        d2 = ensureDeferred(d1)
        self.assertIs(d1, d2)

    def test_willNotAllowNonDeferredOrCoroutine(self) -> None:
        """
        Passing L{ensureDeferred} a non-coroutine and a non-Deferred will
        raise a L{ValueError}.
        """
        with self.assertRaises(defer.NotACoroutineError):
            ensureDeferred('something')

    def test_ensureDeferredCoroutine(self) -> None:
        """
        L{ensureDeferred} will turn a coroutine into a L{Deferred}.
        """

        async def run() -> str:
            d = defer.succeed('foo')
            res = await d
            return res
        r = run()
        self.assertIsInstance(r, types.CoroutineType)
        d = ensureDeferred(r)
        assert_type(d, Deferred[str])
        self.assertIsInstance(d, Deferred)
        res = self.successResultOf(d)
        self.assertEqual(res, 'foo')

    def test_ensureDeferredGenerator(self) -> None:
        """
        L{ensureDeferred} will turn a yield-from coroutine into a L{Deferred}.
        """

        def run() -> Generator[Deferred[str], None, str]:
            d = defer.succeed('foo')
            res = cast(str, (yield from d))
            return res
        r = run()
        self.assertIsInstance(r, types.GeneratorType)
        d = ensureDeferred(r)
        assert_type(d, Deferred[str])
        self.assertIsInstance(d, Deferred)
        res = self.successResultOf(d)
        self.assertEqual(res, 'foo')