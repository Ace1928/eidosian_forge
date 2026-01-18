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
class AlreadyCalledTests(unittest.SynchronousTestCase):

    def setUp(self) -> None:
        self._deferredWasDebugging = defer.getDebugging()
        defer.setDebugging(True)

    def tearDown(self) -> None:
        defer.setDebugging(self._deferredWasDebugging)

    def _callback(self, *args: object, **kwargs: object) -> None:
        pass

    def _errback(self, *args: object, **kwargs: object) -> None:
        pass

    def _call_1(self, d: Deferred[str]) -> None:
        d.callback('hello')

    def _call_2(self, d: Deferred[str]) -> None:
        d.callback('twice')

    def _err_1(self, d: Deferred[str]) -> None:
        d.errback(Failure(RuntimeError()))

    def _err_2(self, d: Deferred[str]) -> None:
        d.errback(Failure(RuntimeError()))

    def testAlreadyCalled_CC(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._call_1(d)
        self.assertRaises(defer.AlreadyCalledError, self._call_2, d)

    def testAlreadyCalled_CE(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._call_1(d)
        self.assertRaises(defer.AlreadyCalledError, self._err_2, d)

    def testAlreadyCalled_EE(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._err_1(d)
        self.assertRaises(defer.AlreadyCalledError, self._err_2, d)

    def testAlreadyCalled_EC(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._err_1(d)
        self.assertRaises(defer.AlreadyCalledError, self._call_2, d)

    def _count(self, linetype: str, func: str, lines: List[str], expected: int) -> None:
        count = 0
        for line in lines:
            if line.startswith(' %s:' % linetype) and line.endswith(' %s' % func):
                count += 1
        self.assertTrue(count == expected)

    def _check(self, e: Exception, caller: str, invoker1: str, invoker2: str) -> None:
        lines = e.args[0].split('\n')
        self._count('C', caller, lines, 1)
        self._count('C', '_call_1', lines, 0)
        self._count('C', '_call_2', lines, 0)
        self._count('C', '_err_1', lines, 0)
        self._count('C', '_err_2', lines, 0)
        self._count('I', invoker1, lines, 1)
        self._count('I', invoker2, lines, 0)

    def testAlreadyCalledDebug_CC(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._call_1(d)
        try:
            self._call_2(d)
        except defer.AlreadyCalledError as e:
            self._check(e, 'testAlreadyCalledDebug_CC', '_call_1', '_call_2')
        else:
            self.fail('second callback failed to raise AlreadyCalledError')

    def testAlreadyCalledDebug_CE(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._call_1(d)
        try:
            self._err_2(d)
        except defer.AlreadyCalledError as e:
            self._check(e, 'testAlreadyCalledDebug_CE', '_call_1', '_err_2')
        else:
            self.fail('second errback failed to raise AlreadyCalledError')

    def testAlreadyCalledDebug_EC(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._err_1(d)
        try:
            self._call_2(d)
        except defer.AlreadyCalledError as e:
            self._check(e, 'testAlreadyCalledDebug_EC', '_err_1', '_call_2')
        else:
            self.fail('second callback failed to raise AlreadyCalledError')

    def testAlreadyCalledDebug_EE(self) -> None:
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._err_1(d)
        try:
            self._err_2(d)
        except defer.AlreadyCalledError as e:
            self._check(e, 'testAlreadyCalledDebug_EE', '_err_1', '_err_2')
        else:
            self.fail('second errback failed to raise AlreadyCalledError')

    def testNoDebugging(self) -> None:
        defer.setDebugging(False)
        d: Deferred[str] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        self._call_1(d)
        try:
            self._call_2(d)
        except defer.AlreadyCalledError as e:
            self.assertFalse(e.args)
        else:
            self.fail('second callback failed to raise AlreadyCalledError')

    def testSwitchDebugging(self) -> None:
        defer.setDebugging(False)
        d: Deferred[None] = Deferred()
        d.addBoth(lambda ign: None)
        defer.setDebugging(True)
        d.callback(None)
        defer.setDebugging(False)
        d = Deferred()
        d.callback(None)
        defer.setDebugging(True)
        d.addBoth(lambda ign: None)