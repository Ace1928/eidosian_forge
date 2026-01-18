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
class TestErrorHolder(TestCase):
    run_test_with = FullStackRunTest

    def makeException(self):
        try:
            raise RuntimeError('danger danger')
        except:
            return sys.exc_info()

    def makePlaceHolder(self, test_id='foo', error=None, short_description=None):
        if error is None:
            error = self.makeException()
        return ErrorHolder(test_id, error, short_description)

    def test_id_comes_from_constructor(self):
        test = ErrorHolder('test id', self.makeException())
        self.assertEqual('test id', test.id())

    def test_shortDescription_is_id(self):
        test = ErrorHolder('test id', self.makeException())
        self.assertEqual(test.id(), test.shortDescription())

    def test_shortDescription_specified(self):
        test = ErrorHolder('test id', self.makeException(), 'description')
        self.assertEqual('description', test.shortDescription())

    def test_counts_as_one_test(self):
        test = self.makePlaceHolder()
        self.assertEqual(1, test.countTestCases())

    def test_str_is_id(self):
        test = self.makePlaceHolder()
        self.assertEqual(test.id(), str(test))

    def test_runs_as_error(self):
        error = self.makeException()
        test = self.makePlaceHolder(error=error)
        result = ExtendedTestResult()
        log = result._events
        test.run(result)
        self.assertEqual([('tags', set(), set()), ('startTest', test), ('addError', test, test._details), ('stopTest', test), ('tags', set(), set())], log)

    def test_call_is_run(self):
        test = self.makePlaceHolder()
        run_log = []
        test.run(LoggingResult(run_log))
        call_log = []
        test(LoggingResult(call_log))
        self.assertEqual(run_log, call_log)

    def test_runs_without_result(self):
        self.makePlaceHolder().run()

    def test_debug(self):
        self.makePlaceHolder().debug()