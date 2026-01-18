from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
class ExpectTests(unittest.TestCase):

    def setUp(self) -> None:
        self.term = helper.ExpectableBuffer()
        self.term.connectionMade()
        self.fs = FakeScheduler()

    def testSimpleString(self) -> None:
        result: list[re.Match[bytes]] = []
        d = self.term.expect(b'hello world', timeout=1, scheduler=self.fs)
        d.addCallback(result.append)
        self.term.write(b'greeting puny earthlings\n')
        self.assertFalse(result)
        self.term.write(b'hello world\n')
        self.assertTrue(result)
        self.assertEqual(result[0].group(), b'hello world')
        self.assertEqual(len(self.fs.calls), 1)
        self.assertFalse(self.fs.calls[0].active())

    def testBrokenUpString(self) -> None:
        result: list[re.Match[bytes]] = []
        d = self.term.expect(b'hello world')
        d.addCallback(result.append)
        self.assertFalse(result)
        self.term.write(b'hello ')
        self.assertFalse(result)
        self.term.write(b'worl')
        self.assertFalse(result)
        self.term.write(b'd')
        self.assertTrue(result)
        self.assertEqual(result[0].group(), b'hello world')

    def testMultiple(self) -> None:
        result: list[re.Match[bytes]] = []
        d1 = self.term.expect(b'hello ')
        d1.addCallback(result.append)
        d2 = self.term.expect(b'world')
        d2.addCallback(result.append)
        self.assertFalse(result)
        self.term.write(b'hello')
        self.assertFalse(result)
        self.term.write(b' ')
        self.assertEqual(len(result), 1)
        self.term.write(b'world')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].group(), b'hello ')
        self.assertEqual(result[1].group(), b'world')

    def testSynchronous(self) -> None:
        self.term.write(b'hello world')
        result: list[re.Match[bytes]] = []
        d = self.term.expect(b'hello world')
        d.addCallback(result.append)
        self.assertTrue(result)
        self.assertEqual(result[0].group(), b'hello world')

    def testMultipleSynchronous(self) -> None:
        self.term.write(b'goodbye world')
        result: list[re.Match[bytes]] = []
        d1 = self.term.expect(b'bye')
        d1.addCallback(result.append)
        d2 = self.term.expect(b'world')
        d2.addCallback(result.append)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].group(), b'bye')
        self.assertEqual(result[1].group(), b'world')

    def _cbTestTimeoutFailure(self, res: failure.Failure) -> None:
        self.assertTrue(hasattr(res, 'type'))
        self.assertEqual(res.type, helper.ExpectationTimeout)

    def testTimeoutFailure(self) -> None:
        d = self.term.expect(b'hello world', timeout=1, scheduler=self.fs)
        d.addBoth(self._cbTestTimeoutFailure)
        self.fs.calls[0].call()

    def testOverlappingTimeout(self) -> None:
        self.term.write(b'not zoomtastic')
        result: list[re.Match[bytes]] = []
        d1 = self.term.expect(b'hello world', timeout=1, scheduler=self.fs)
        d1.addBoth(self._cbTestTimeoutFailure)
        d2 = self.term.expect(b'zoom')
        d2.addCallback(result.append)
        self.fs.calls[0].call()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].group(), b'zoom')