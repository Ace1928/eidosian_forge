from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
class ForceGarbageCollectionTests(unittest.SynchronousTestCase):
    """
    Tests for the --force-gc option.
    """

    def setUp(self) -> None:
        self.config = trial.Options()
        self.log: list[str] = []
        self.patch(gc, 'collect', self.collect)
        test = pyunit.FunctionTestCase(self.simpleTest)
        self.test = TestSuite([test, test])

    def simpleTest(self) -> None:
        """
        A simple test method that records that it was run.
        """
        self.log.append('test')

    def collect(self) -> None:
        """
        A replacement for gc.collect that logs calls to itself.
        """
        self.log.append('collect')

    def makeRunner(self) -> _Runner:
        """
        Return a L{TrialRunner} object that is safe to use in tests.
        """
        runner = trial._makeRunner(self.config)
        runner.stream = StringIO()
        return runner

    def test_forceGc(self) -> None:
        """
        Passing the --force-gc option to the trial script forces the garbage
        collector to run before and after each test.
        """
        self.config['force-gc'] = True
        self.config.postOptions()
        runner = self.makeRunner()
        runner.run(self.test)
        self.assertEqual(self.log, ['collect', 'test', 'collect', 'collect', 'test', 'collect'])

    def test_unforceGc(self) -> None:
        """
        By default, no garbage collection is forced.
        """
        self.config.postOptions()
        runner = self.makeRunner()
        runner.run(self.test)
        self.assertEqual(self.log, ['test', 'test'])