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
class OptionsTests(unittest.TestCase):
    """
    Tests for L{trial.Options}.
    """

    def setUp(self) -> None:
        """
        Build an L{Options} object to be used in the tests.
        """
        self.options = trial.Options()

    def test_getWorkerArguments(self) -> None:
        """
        C{_getWorkerArguments} discards options like C{random} as they only
        matter in the manager, and forwards options like C{recursionlimit} or
        C{disablegc}.
        """
        self.addCleanup(sys.setrecursionlimit, sys.getrecursionlimit())
        if gc.isenabled():
            self.addCleanup(gc.enable)
        self.options.parseOptions(['--recursionlimit', '2000', '--random', '4', '--disablegc'])
        args = self.options._getWorkerArguments()
        self.assertIn('--disablegc', args)
        args.remove('--disablegc')
        self.assertEqual(['--recursionlimit', '2000'], args)

    def test_jobsConflictWithDebug(self) -> None:
        """
        C{parseOptions} raises a C{UsageError} when C{--debug} is passed along
        C{--jobs} as it's not supported yet.

        @see: U{http://twistedmatrix.com/trac/ticket/5825}
        """
        error = self.assertRaises(UsageError, self.options.parseOptions, ['--jobs', '4', '--debug'])
        self.assertEqual("You can't specify --debug when using --jobs", str(error))

    def test_jobsConflictWithProfile(self) -> None:
        """
        C{parseOptions} raises a C{UsageError} when C{--profile} is passed
        along C{--jobs} as it's not supported yet.

        @see: U{http://twistedmatrix.com/trac/ticket/5827}
        """
        error = self.assertRaises(UsageError, self.options.parseOptions, ['--jobs', '4', '--profile'])
        self.assertEqual("You can't specify --profile when using --jobs", str(error))

    def test_jobsConflictWithDebugStackTraces(self) -> None:
        """
        C{parseOptions} raises a C{UsageError} when C{--debug-stacktraces} is
        passed along C{--jobs} as it's not supported yet.

        @see: U{http://twistedmatrix.com/trac/ticket/5826}
        """
        error = self.assertRaises(UsageError, self.options.parseOptions, ['--jobs', '4', '--debug-stacktraces'])
        self.assertEqual("You can't specify --debug-stacktraces when using --jobs", str(error))

    def test_orderConflictWithRandom(self) -> None:
        """
        C{parseOptions} raises a C{UsageError} when C{--order} is passed along
        with C{--random}.
        """
        error = self.assertRaises(UsageError, self.options.parseOptions, ['--order', 'alphabetical', '--random', '1234'])
        self.assertEqual("You can't specify --random when using --order", str(error))