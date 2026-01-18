from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestRunTestUsage(TestCase):

    def test_last_resort_in_place(self):

        class TestBase(TestCase):

            def test_base_exception(self):
                raise SystemExit(0)
        result = ExtendedTestResult()
        test = TestBase('test_base_exception')
        self.assertRaises(SystemExit, test.run, result)
        self.assertFalse(result.wasSuccessful())