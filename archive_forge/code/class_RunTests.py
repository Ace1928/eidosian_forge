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
class RunTests(unittest.TestCase):
    """
    Tests for the L{run} function.
    """

    def setUp(self) -> None:
        self.patch(trial.Options, 'parseOptions', lambda self: None)

    def test_debuggerNotFound(self) -> None:
        """
        When a debugger is not found, an error message is printed to the user.

        """

        def _makeRunner(*args: object, **kwargs: object) -> None:
            raise trial._DebuggerNotFound('foo')
        self.patch(trial, '_makeRunner', _makeRunner)
        try:
            trial.run()
        except SystemExit as e:
            self.assertIn('foo', str(e))
        else:
            self.fail('Should have exited due to non-existent debugger!')