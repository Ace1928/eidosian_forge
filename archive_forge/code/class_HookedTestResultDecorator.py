import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class HookedTestResultDecorator(TestResultDecorator):
    """A TestResult which calls a hook on every event."""

    def __init__(self, decorated):
        self.super = super()
        self.super.__init__(decorated)

    def startTest(self, test):
        self._before_event()
        return self.super.startTest(test)

    def startTestRun(self):
        self._before_event()
        return self.super.startTestRun()

    def stopTest(self, test):
        self._before_event()
        return self.super.stopTest(test)

    def stopTestRun(self):
        self._before_event()
        return self.super.stopTestRun()

    def addError(self, test, err=None, details=None):
        self._before_event()
        return self.super.addError(test, err, details=details)

    def addFailure(self, test, err=None, details=None):
        self._before_event()
        return self.super.addFailure(test, err, details=details)

    def addSuccess(self, test, details=None):
        self._before_event()
        return self.super.addSuccess(test, details=details)

    def addSkip(self, test, reason=None, details=None):
        self._before_event()
        return self.super.addSkip(test, reason, details=details)

    def addExpectedFailure(self, test, err=None, details=None):
        self._before_event()
        return self.super.addExpectedFailure(test, err, details=details)

    def addUnexpectedSuccess(self, test, details=None):
        self._before_event()
        return self.super.addUnexpectedSuccess(test, details=details)

    def progress(self, offset, whence):
        self._before_event()
        return self.super.progress(offset, whence)

    def wasSuccessful(self):
        self._before_event()
        return self.super.wasSuccessful()

    @property
    def shouldStop(self):
        self._before_event()
        return self.super.shouldStop

    def stop(self):
        self._before_event()
        return self.super.stop()

    def time(self, a_datetime):
        self._before_event()
        return self.super.time(a_datetime)