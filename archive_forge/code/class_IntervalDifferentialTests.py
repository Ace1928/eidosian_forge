import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
class IntervalDifferentialTests(TestCase):

    def testDefault(self):
        d = iter(util.IntervalDifferential([], 10))
        for i in range(100):
            self.assertEqual(next(d), (10, None))

    def testSingle(self):
        d = iter(util.IntervalDifferential([5], 10))
        for i in range(100):
            self.assertEqual(next(d), (5, 0))

    def testPair(self):
        d = iter(util.IntervalDifferential([5, 7], 10))
        for i in range(100):
            self.assertEqual(next(d), (5, 0))
            self.assertEqual(next(d), (2, 1))
            self.assertEqual(next(d), (3, 0))
            self.assertEqual(next(d), (4, 1))
            self.assertEqual(next(d), (1, 0))
            self.assertEqual(next(d), (5, 0))
            self.assertEqual(next(d), (1, 1))
            self.assertEqual(next(d), (4, 0))
            self.assertEqual(next(d), (3, 1))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (5, 0))
            self.assertEqual(next(d), (0, 1))

    def testTriple(self):
        d = iter(util.IntervalDifferential([2, 4, 5], 10))
        for i in range(100):
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (0, 1))
            self.assertEqual(next(d), (1, 2))
            self.assertEqual(next(d), (1, 0))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (0, 1))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (0, 2))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (0, 1))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (1, 2))
            self.assertEqual(next(d), (1, 0))
            self.assertEqual(next(d), (0, 1))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (2, 0))
            self.assertEqual(next(d), (0, 1))
            self.assertEqual(next(d), (0, 2))

    def testInsert(self):
        d = iter(util.IntervalDifferential([], 10))
        self.assertEqual(next(d), (10, None))
        d.addInterval(3)
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (3, 0))
        d.addInterval(6)
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (0, 1))

    def testRemove(self):
        d = iter(util.IntervalDifferential([3, 5], 10))
        self.assertEqual(next(d), (3, 0))
        self.assertEqual(next(d), (2, 1))
        self.assertEqual(next(d), (1, 0))
        d.removeInterval(3)
        self.assertEqual(next(d), (4, 0))
        self.assertEqual(next(d), (5, 0))
        d.removeInterval(5)
        self.assertEqual(next(d), (10, None))
        self.assertRaises(ValueError, d.removeInterval, 10)