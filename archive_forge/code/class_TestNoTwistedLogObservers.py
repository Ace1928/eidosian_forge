import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestNoTwistedLogObservers(NeedsTwistedTestCase):
    """Tests for _NoTwistedLogObservers."""

    def _get_logged_messages(self, function, *args, **kwargs):
        """Run ``function`` and return ``(ret, logged_messages)``."""
        messages = []
        publisher, _ = _get_global_publisher_and_observers()
        publisher.addObserver(messages.append)
        try:
            ret = function(*args, **kwargs)
        finally:
            publisher.removeObserver(messages.append)
        return (ret, messages)

    def test_default(self):

        class SomeTest(TestCase):

            def test_something(self):
                log.msg('foo')
        _, messages = self._get_logged_messages(SomeTest('test_something').run)
        self.assertThat(messages, MatchesListwise([ContainsDict({'message': Equals(('foo',))})]))

    def test_nothing_logged(self):
        from testtools.twistedsupport._runtest import _NoTwistedLogObservers

        class SomeTest(TestCase):

            def test_something(self):
                self.useFixture(_NoTwistedLogObservers())
                log.msg('foo')
        _, messages = self._get_logged_messages(SomeTest('test_something').run)
        self.assertThat(messages, Equals([]))

    def test_logging_restored(self):
        from testtools.twistedsupport._runtest import _NoTwistedLogObservers

        class SomeTest(TestCase):

            def test_something(self):
                self.useFixture(_NoTwistedLogObservers())
                log.msg('foo')

        def run_then_log():
            SomeTest('test_something').run()
            log.msg('bar')
        _, messages = self._get_logged_messages(run_then_log)
        self.assertThat(messages, MatchesListwise([ContainsDict({'message': Equals(('bar',))})]))