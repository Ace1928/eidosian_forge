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
class TestSetupTearDown(TestCase):
    run_test_with = FullStackRunTest

    def test_setUpCalledTwice(self):

        class CallsTooMuch(TestCase):

            def test_method(self):
                self.setUp()
        result = unittest.TestResult()
        CallsTooMuch('test_method').run(result)
        self.assertThat(result.errors, HasLength(1))
        self.assertThat(result.errors[0][1], DocTestMatches('...ValueError...File...testtools/tests/test_testcase.py...', ELLIPSIS))

    def test_setUpNotCalled(self):

        class DoesnotcallsetUp(TestCase):

            def setUp(self):
                pass

            def test_method(self):
                pass
        result = unittest.TestResult()
        DoesnotcallsetUp('test_method').run(result)
        self.assertThat(result.errors, HasLength(1))
        self.assertThat(result.errors[0][1], DocTestMatches('...ValueError...File...testtools/tests/test_testcase.py...', ELLIPSIS))

    def test_tearDownCalledTwice(self):

        class CallsTooMuch(TestCase):

            def test_method(self):
                self.tearDown()
        result = unittest.TestResult()
        CallsTooMuch('test_method').run(result)
        self.assertThat(result.errors, HasLength(1))
        self.assertThat(result.errors[0][1], DocTestMatches('...ValueError...File...testtools/tests/test_testcase.py...', ELLIPSIS))

    def test_tearDownNotCalled(self):

        class DoesnotcalltearDown(TestCase):

            def test_method(self):
                pass

            def tearDown(self):
                pass
        result = unittest.TestResult()
        DoesnotcalltearDown('test_method').run(result)
        self.assertThat(result.errors, HasLength(1))
        self.assertThat(result.errors[0][1], DocTestMatches('...ValueError...File...testtools/tests/test_testcase.py...', ELLIPSIS))