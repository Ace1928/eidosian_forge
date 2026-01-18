import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestDocTestSuiteIsolation(tests.TestCase):
    """Test that `tests.DocTestSuite` isolates doc tests from os.environ.

    Since tests.TestCase alreay provides an isolation from os.environ, we use
    the clean environment as a base for testing. To precisely capture the
    isolation provided by tests.DocTestSuite, we use doctest.DocTestSuite to
    compare against.

    We want to make sure `tests.DocTestSuite` respect `tests.isolated_environ`,
    not `os.environ` so each test overrides it to suit its needs.

    """

    def get_doctest_suite_for_string(self, klass, string):

        class Finder(doctest.DocTestFinder):

            def find(*args, **kwargs):
                test = doctest.DocTestParser().get_doctest(string, {}, 'foo', 'foo.py', 0)
                return [test]
        suite = klass(test_finder=Finder())
        return suite

    def run_doctest_suite_for_string(self, klass, string):
        suite = self.get_doctest_suite_for_string(klass, string)
        output = StringIO()
        result = tests.TextTestResult(output, 0, 1)
        suite.run(result)
        return (result, output)

    def assertDocTestStringSucceds(self, klass, string):
        result, output = self.run_doctest_suite_for_string(klass, string)
        if not result.wasStrictlySuccessful():
            self.fail(output.getvalue())

    def assertDocTestStringFails(self, klass, string):
        result, output = self.run_doctest_suite_for_string(klass, string)
        if result.wasStrictlySuccessful():
            self.fail(output.getvalue())

    def test_injected_variable(self):
        self.overrideAttr(tests, 'isolated_environ', {'LINES': '42'})
        test = "\n            >>> import os\n            >>> os.environ['LINES']\n            '42'\n            "
        self.assertDocTestStringFails(doctest.DocTestSuite, test)
        self.assertDocTestStringSucceds(tests.IsolatedDocTestSuite, test)

    def test_deleted_variable(self):
        self.overrideAttr(tests, 'isolated_environ', {'LINES': None})
        test = "\n            >>> import os\n            >>> os.environ.get('LINES')\n            "
        self.assertDocTestStringFails(doctest.DocTestSuite, test)
        self.assertDocTestStringSucceds(tests.IsolatedDocTestSuite, test)