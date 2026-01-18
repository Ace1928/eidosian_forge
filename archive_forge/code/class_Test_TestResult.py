import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class Test_TestResult(unittest.TestCase):

    def test_init(self):
        result = unittest.TestResult()
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 0)
        self.assertEqual(result.shouldStop, False)
        self.assertIsNone(result._stdout_buffer)
        self.assertIsNone(result._stderr_buffer)

    def test_stop(self):
        result = unittest.TestResult()
        result.stop()
        self.assertEqual(result.shouldStop, True)

    def test_startTest(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')
        result = unittest.TestResult()
        result.startTest(test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)
        result.stopTest(test)

    def test_stopTest(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')
        result = unittest.TestResult()
        result.startTest(test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)
        result.stopTest(test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)

    def test_startTestRun_stopTestRun(self):
        result = unittest.TestResult()
        result.startTestRun()
        result.stopTestRun()

    def test_addSuccess(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')
        result = unittest.TestResult()
        result.startTest(test)
        result.addSuccess(test)
        result.stopTest(test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)

    def test_addFailure(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')
        try:
            test.fail('foo')
        except:
            exc_info_tuple = sys.exc_info()
        result = unittest.TestResult()
        result.startTest(test)
        result.addFailure(test, exc_info_tuple)
        result.stopTest(test)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)
        test_case, formatted_exc = result.failures[0]
        self.assertIs(test_case, test)
        self.assertIsInstance(formatted_exc, str)

    def test_addFailure_filter_traceback_frames(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')

        def get_exc_info():
            try:
                test.fail('foo')
            except:
                return sys.exc_info()
        exc_info_tuple = get_exc_info()
        full_exc = traceback.format_exception(*exc_info_tuple)
        result = unittest.TestResult()
        result.startTest(test)
        result.addFailure(test, exc_info_tuple)
        result.stopTest(test)
        formatted_exc = result.failures[0][1]
        dropped = [l for l in full_exc if l not in formatted_exc]
        self.assertEqual(len(dropped), 1)
        self.assertIn('raise self.failureException(msg)', dropped[0])

    def test_addFailure_filter_traceback_frames_context(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')

        def get_exc_info():
            try:
                try:
                    test.fail('foo')
                except:
                    raise ValueError(42)
            except:
                return sys.exc_info()
        exc_info_tuple = get_exc_info()
        full_exc = traceback.format_exception(*exc_info_tuple)
        result = unittest.TestResult()
        result.startTest(test)
        result.addFailure(test, exc_info_tuple)
        result.stopTest(test)
        formatted_exc = result.failures[0][1]
        dropped = [l for l in full_exc if l not in formatted_exc]
        self.assertEqual(len(dropped), 1)
        self.assertIn('raise self.failureException(msg)', dropped[0])

    def test_addFailure_filter_traceback_frames_chained_exception_self_loop(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass

        def get_exc_info():
            try:
                loop = Exception('Loop')
                loop.__cause__ = loop
                loop.__context__ = loop
                raise loop
            except:
                return sys.exc_info()
        exc_info_tuple = get_exc_info()
        test = Foo('test_1')
        result = unittest.TestResult()
        result.startTest(test)
        result.addFailure(test, exc_info_tuple)
        result.stopTest(test)
        formatted_exc = result.failures[0][1]
        self.assertEqual(formatted_exc.count('Exception: Loop\n'), 1)

    def test_addFailure_filter_traceback_frames_chained_exception_cycle(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass

        def get_exc_info():
            try:
                A, B, C = (Exception('A'), Exception('B'), Exception('C'))
                edges = [(C, B), (B, A), (A, C)]
                for ex1, ex2 in edges:
                    ex1.__cause__ = ex2
                    ex2.__context__ = ex1
                raise C
            except:
                return sys.exc_info()
        exc_info_tuple = get_exc_info()
        test = Foo('test_1')
        result = unittest.TestResult()
        result.startTest(test)
        result.addFailure(test, exc_info_tuple)
        result.stopTest(test)
        formatted_exc = result.failures[0][1]
        self.assertEqual(formatted_exc.count('Exception: A\n'), 1)
        self.assertEqual(formatted_exc.count('Exception: B\n'), 1)
        self.assertEqual(formatted_exc.count('Exception: C\n'), 1)

    def test_addError(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                pass
        test = Foo('test_1')
        try:
            raise TypeError()
        except:
            exc_info_tuple = sys.exc_info()
        result = unittest.TestResult()
        result.startTest(test)
        result.addError(test, exc_info_tuple)
        result.stopTest(test)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)
        test_case, formatted_exc = result.errors[0]
        self.assertIs(test_case, test)
        self.assertIsInstance(formatted_exc, str)

    def test_addError_locals(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                1 / 0
        test = Foo('test_1')
        result = unittest.TestResult()
        result.tb_locals = True
        unittest.result.traceback = MockTraceback
        self.addCleanup(restore_traceback)
        result.startTestRun()
        test.run(result)
        result.stopTestRun()
        self.assertEqual(len(result.errors), 1)
        test_case, formatted_exc = result.errors[0]
        self.assertEqual('A tracebacklocals', formatted_exc)

    def test_addSubTest(self):

        class Foo(unittest.TestCase):

            def test_1(self):
                nonlocal subtest
                with self.subTest(foo=1):
                    subtest = self._subtest
                    try:
                        1 / 0
                    except ZeroDivisionError:
                        exc_info_tuple = sys.exc_info()
                    result.addSubTest(test, subtest, exc_info_tuple)
                    self.fail('some recognizable failure')
        subtest = None
        test = Foo('test_1')
        result = unittest.TestResult()
        test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.shouldStop, False)
        test_case, formatted_exc = result.errors[0]
        self.assertIs(test_case, subtest)
        self.assertIn('ZeroDivisionError', formatted_exc)
        test_case, formatted_exc = result.failures[0]
        self.assertIs(test_case, subtest)
        self.assertIn('some recognizable failure', formatted_exc)

    def testStackFrameTrimming(self):

        class Frame(object):

            class tb_frame(object):
                f_globals = {}
        result = unittest.TestResult()
        self.assertFalse(result._is_relevant_tb_level(Frame))
        Frame.tb_frame.f_globals['__unittest'] = True
        self.assertTrue(result._is_relevant_tb_level(Frame))

    def testFailFast(self):
        result = unittest.TestResult()
        result._exc_info_to_string = lambda *_: ''
        result.failfast = True
        result.addError(None, None)
        self.assertTrue(result.shouldStop)
        result = unittest.TestResult()
        result._exc_info_to_string = lambda *_: ''
        result.failfast = True
        result.addFailure(None, None)
        self.assertTrue(result.shouldStop)
        result = unittest.TestResult()
        result._exc_info_to_string = lambda *_: ''
        result.failfast = True
        result.addUnexpectedSuccess(None)
        self.assertTrue(result.shouldStop)

    def testFailFastSetByRunner(self):
        stream = BufferedWriter()
        runner = unittest.TextTestRunner(stream=stream, failfast=True)

        def test(result):
            self.assertTrue(result.failfast)
        result = runner.run(test)
        stream.flush()
        self.assertTrue(stream.getvalue().endswith('\n\nOK\n'))