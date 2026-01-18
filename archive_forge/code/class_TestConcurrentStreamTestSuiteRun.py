import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
class TestConcurrentStreamTestSuiteRun(TestCase):

    def test_trivial(self):
        result = LoggingStream()
        test1 = Sample('test_method1')
        test2 = Sample('test_method2')

        def cases():
            return [(test1, '0'), (test2, '1')]
        suite = ConcurrentStreamTestSuite(cases)
        suite.run(result)

        def freeze(set_or_none):
            if set_or_none is None:
                return set_or_none
            return frozenset(set_or_none)
        self.assertEqual({('status', 'testtools.tests.test_testsuite.Sample.test_method1', 'inprogress', None, True, None, None, False, None, '0', None), ('status', 'testtools.tests.test_testsuite.Sample.test_method1', 'success', frozenset(), True, None, None, False, None, '0', None), ('status', 'testtools.tests.test_testsuite.Sample.test_method2', 'inprogress', None, True, None, None, False, None, '1', None), ('status', 'testtools.tests.test_testsuite.Sample.test_method2', 'success', frozenset(), True, None, None, False, None, '1', None)}, {event[0:3] + (freeze(event[3]),) + event[4:10] + (None,) for event in result._events})

    def test_broken_runner(self):

        class BrokenTest:

            def __call__(self):
                pass

            def run(self):
                pass
        result = LoggingStream()

        def cases():
            return [(BrokenTest(), '0')]
        suite = ConcurrentStreamTestSuite(cases)
        suite.run(result)
        events = result._events
        self.assertEqual(events[1][6].decode('utf8'), 'Traceback (most recent call last):\n')
        self.assertThat(events[2][6].decode('utf8'), DocTestMatches('  File "...testtools/testsuite.py", line ..., in _run_test\n    test.run(process_result)...\n', doctest.ELLIPSIS))
        self.assertThat(events[3][6].decode('utf8'), DocTestMatches('TypeError: ...run() takes ...1 ...argument...2...given...\n', doctest.ELLIPSIS))
        events = [event[0:10] + (None,) for event in events]
        events[1] = events[1][:6] + (None,) + events[1][7:]
        events[2] = events[2][:6] + (None,) + events[2][7:]
        events[3] = events[3][:6] + (None,) + events[3][7:]
        self.assertEqual([('status', "broken-runner-'0'", 'inprogress', None, True, None, None, False, None, '0', None), ('status', "broken-runner-'0'", None, None, True, 'traceback', None, False, 'text/x-traceback; charset="utf8"; language="python"', '0', None), ('status', "broken-runner-'0'", None, None, True, 'traceback', None, False, 'text/x-traceback; charset="utf8"; language="python"', '0', None), ('status', "broken-runner-'0'", None, None, True, 'traceback', None, True, 'text/x-traceback; charset="utf8"; language="python"', '0', None), ('status', "broken-runner-'0'", 'fail', set(), True, None, None, False, None, '0', None)], events)

    def split_suite(self, suite):
        tests = list(enumerate(iterate_tests(suite)))
        return [(test, str(pos)) for pos, test in tests]

    def test_setupclass_skip(self):

        class Skips(TestCase):

            @classmethod
            def setUpClass(cls):
                raise cls.skipException('foo')

            def test_notrun(self):
                pass
        suite = unittest.TestSuite([Skips('test_notrun')])
        log = []
        result = LoggingResult(log)
        suite.run(result)
        self.assertEqual(['addSkip'], [item[0] for item in log])

    def test_setupclass_upcall(self):

        class Simples(TestCase):

            @classmethod
            def setUpClass(cls):
                super().setUpClass()

            def test_simple(self):
                pass
        suite = unittest.TestSuite([Simples('test_simple')])
        log = []
        result = LoggingResult(log)
        suite.run(result)
        self.assertEqual(['startTest', 'addSuccess', 'stopTest'], [item[0] for item in log])