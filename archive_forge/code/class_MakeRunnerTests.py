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
class MakeRunnerTests(unittest.TestCase):
    """
    Tests for the L{_makeRunner} helper.
    """

    def setUp(self) -> None:
        self.options = trial.Options()

    def test_jobs(self) -> None:
        """
        L{_makeRunner} returns a L{DistTrialRunner} instance when the C{--jobs}
        option is passed.  The L{DistTrialRunner} knows how many workers to
        run and the C{workerArguments} to pass to them.
        """
        self.options.parseOptions(['--jobs', '4', '--force-gc'])
        runner = trial._makeRunner(self.options)
        assert isinstance(runner, DistTrialRunner)
        self.assertIsInstance(runner, DistTrialRunner)
        self.assertEqual(4, runner._maxWorkers)
        self.assertEqual(['--force-gc'], runner._workerArguments)

    def test_dryRunWithJobs(self) -> None:
        """
        L{_makeRunner} returns a L{TrialRunner} instance in C{DRY_RUN} mode
        when the C{--dry-run} option is passed, even if C{--jobs} is set.
        """
        self.options.parseOptions(['--jobs', '4', '--dry-run'])
        runner = trial._makeRunner(self.options)
        assert isinstance(runner, TrialRunner)
        self.assertIsInstance(runner, TrialRunner)
        self.assertEqual(TrialRunner.DRY_RUN, runner.mode)

    def test_DebuggerNotFound(self) -> None:
        namedAny = trial.reflect.namedAny

        def namedAnyExceptdoNotFind(fqn: str) -> object:
            if fqn == 'doNotFind':
                raise trial.reflect.ModuleNotFound(fqn)
            return namedAny(fqn)
        self.patch(trial.reflect, 'namedAny', namedAnyExceptdoNotFind)
        options = trial.Options()
        options.parseOptions(['--debug', '--debugger', 'doNotFind'])
        self.assertRaises(trial._DebuggerNotFound, trial._makeRunner, options)

    def test_exitfirst(self) -> None:
        """
        Passing C{--exitfirst} wraps the reporter with a
        L{reporter._ExitWrapper} that stops on any non-success.
        """
        self.options.parseOptions(['--exitfirst'])
        runner = trial._makeRunner(self.options)
        assert isinstance(runner, TrialRunner)
        self.assertTrue(runner._exitFirst)