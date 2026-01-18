from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
class TestTestCaseSupportForRunTest(TestCase):

    def test_pass_custom_run_test(self):

        class SomeCase(TestCase):

            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo', runTest=CustomRunTest)
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(CustomRunTest.marker))

    def test_default_is_runTest_class_variable(self):

        class SomeCase(TestCase):
            run_tests_with = CustomRunTest

            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo')
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(CustomRunTest.marker))

    def test_constructor_argument_overrides_class_variable(self):
        marker = object()

        class DifferentRunTest(RunTest):

            def run(self, result=None):
                return marker

        class SomeCase(TestCase):
            run_tests_with = CustomRunTest

            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo', runTest=DifferentRunTest)
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(marker))

    def test_decorator_for_run_test(self):

        class SomeCase(TestCase):

            @run_test_with(CustomRunTest)
            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo')
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(CustomRunTest.marker))

    def test_extended_decorator_for_run_test(self):
        marker = object()

        class FooRunTest(RunTest):

            def __init__(self, case, handlers=None, bar=None):
                super().__init__(case, handlers)
                self.bar = bar

            def run(self, result=None):
                return self.bar

        class SomeCase(TestCase):

            @run_test_with(FooRunTest, bar=marker)
            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo')
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(marker))

    def test_works_as_inner_decorator(self):

        def wrapped(function):
            """Silly, trivial decorator."""

            def decorated(*args, **kwargs):
                return function(*args, **kwargs)
            decorated.__name__ = function.__name__
            decorated.__dict__.update(function.__dict__)
            return decorated

        class SomeCase(TestCase):

            @wrapped
            @run_test_with(CustomRunTest)
            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo')
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(CustomRunTest.marker))

    def test_constructor_overrides_decorator(self):
        marker = object()

        class DifferentRunTest(RunTest):

            def run(self, result=None):
                return marker

        class SomeCase(TestCase):

            @run_test_with(CustomRunTest)
            def test_foo(self):
                pass
        result = TestResult()
        case = SomeCase('test_foo', runTest=DifferentRunTest)
        from_run_test = case.run(result)
        self.assertThat(from_run_test, Is(marker))