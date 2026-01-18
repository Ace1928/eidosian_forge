import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestAssertFailsWith(NeedsTwistedTestCase):
    """Tests for `assert_fails_with`."""
    if SynchronousDeferredRunTest is not None:
        run_tests_with = SynchronousDeferredRunTest

    def test_assert_fails_with_success(self):
        marker = object()
        d = assert_fails_with(defer.succeed(marker), RuntimeError)

        def check_result(failure):
            failure.trap(self.failureException)
            self.assertThat(str(failure.value), Equals(f'RuntimeError not raised ({marker!r} returned)'))
        d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
        return d

    def test_assert_fails_with_success_multiple_types(self):
        marker = object()
        d = assert_fails_with(defer.succeed(marker), RuntimeError, ZeroDivisionError)

        def check_result(failure):
            failure.trap(self.failureException)
            self.assertThat(str(failure.value), Equals('RuntimeError, ZeroDivisionError not raised (%r returned)' % (marker,)))
        d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
        return d

    def test_assert_fails_with_wrong_exception(self):
        d = assert_fails_with(defer.maybeDeferred(lambda: 1 / 0), RuntimeError, KeyboardInterrupt)

        def check_result(failure):
            failure.trap(self.failureException)
            lines = str(failure.value).splitlines()
            self.assertThat(lines[:2], Equals(['ZeroDivisionError raised instead of RuntimeError, KeyboardInterrupt:', ' Traceback (most recent call last):']))
        d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
        return d

    def test_assert_fails_with_expected_exception(self):
        try:
            1 / 0
        except ZeroDivisionError:
            f = failure.Failure()
        d = assert_fails_with(defer.fail(f), ZeroDivisionError)
        return d.addCallback(self.assertThat, Equals(f.value))

    def test_custom_failure_exception(self):

        class CustomException(Exception):
            pass
        marker = object()
        d = assert_fails_with(defer.succeed(marker), RuntimeError, failureException=CustomException)

        def check_result(failure):
            failure.trap(CustomException)
            self.assertThat(str(failure.value), Equals(f'RuntimeError not raised ({marker!r} returned)'))
        return d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)