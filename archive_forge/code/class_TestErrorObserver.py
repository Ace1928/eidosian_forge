import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestErrorObserver(NeedsTwistedTestCase):
    """Tests for _ErrorObserver."""

    def test_captures_errors(self):
        from testtools.twistedsupport._runtest import _ErrorObserver, _LogObserver, _NoTwistedLogObservers
        log_observer = _LogObserver()
        error_observer = _ErrorObserver(log_observer)
        exception = ValueError('bar')

        class SomeTest(TestCase):

            def test_something(self):
                self.useFixture(_NoTwistedLogObservers())
                self.useFixture(error_observer)
                log.msg('foo')
                log.err(exception)
        SomeTest('test_something').run()
        self.assertThat(error_observer.flush_logged_errors(), MatchesListwise([AfterPreprocessing(lambda x: x.value, Equals(exception))]))