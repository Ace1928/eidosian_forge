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
class CoverageTests(unittest.SynchronousTestCase):
    """
    Tests for the I{coverage} option.
    """
    if getattr(sys, 'gettrace', None) is None:
        skip = 'Cannot test trace hook installation without inspection API.'

    def setUp(self) -> None:
        """
        Arrange for the current trace hook to be restored when the
        test is complete.
        """
        self.addCleanup(sys.settrace, sys.gettrace())

    def test_tracerInstalled(self) -> None:
        """
        L{trial.Options} handles C{"--coverage"} by installing a trace
        hook to record coverage information.
        """
        options = trial.Options()
        options.parseOptions(['--coverage'])
        assert options.tracer is not None
        self.assertEqual(sys.gettrace(), options.tracer.globaltrace)

    def test_coverdirDefault(self) -> None:
        """
        L{trial.Options.coverdir} returns a L{FilePath} based on the default
        for the I{temp-directory} option if that option is not specified.
        """
        options = trial.Options()
        self.assertEqual(options.coverdir(), FilePath('.').descendant([options['temp-directory'], 'coverage']))

    def test_coverdirOverridden(self) -> None:
        """
        If a value is specified for the I{temp-directory} option,
        L{trial.Options.coverdir} returns a child of that path.
        """
        path = self.mktemp()
        options = trial.Options()
        options.parseOptions(['--temp-directory', path])
        self.assertEqual(options.coverdir(), FilePath(path).child('coverage'))