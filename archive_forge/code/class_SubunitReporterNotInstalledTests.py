import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
class SubunitReporterNotInstalledTests(unittest.SynchronousTestCase):
    """
    Test behaviour when the subunit reporter is not installed.
    """

    def test_subunitNotInstalled(self):
        """
        If subunit is not installed, TestProtocolClient will be None, and
        SubunitReporter will raise an error when you try to construct it.
        """
        stream = StringIO()
        self.patch(reporter, 'TestProtocolClient', None)
        e = self.assertRaises(Exception, reporter.SubunitReporter, stream)
        self.assertEqual('Subunit not available', str(e))