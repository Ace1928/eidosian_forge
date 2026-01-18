import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TestResultDecorator:
    """General pass-through decorator.

    This provides a base that other TestResults can inherit from to
    gain basic forwarding functionality. It also takes care of
    handling the case where the target doesn't support newer methods
    or features by degrading them.
    """

    def __init__(self, decorated):
        """Create a TestResultDecorator forwarding to decorated."""
        self.decorated = testtools.ExtendedToOriginalDecorator(decorated)

    def startTest(self, test):
        return self.decorated.startTest(test)

    def startTestRun(self):
        return self.decorated.startTestRun()

    def stopTest(self, test):
        return self.decorated.stopTest(test)

    def stopTestRun(self):
        return self.decorated.stopTestRun()

    def addError(self, test, err=None, details=None):
        return self.decorated.addError(test, err, details=details)

    def addFailure(self, test, err=None, details=None):
        return self.decorated.addFailure(test, err, details=details)

    def addSuccess(self, test, details=None):
        return self.decorated.addSuccess(test, details=details)

    def addSkip(self, test, reason=None, details=None):
        return self.decorated.addSkip(test, reason, details=details)

    def addExpectedFailure(self, test, err=None, details=None):
        return self.decorated.addExpectedFailure(test, err, details=details)

    def addUnexpectedSuccess(self, test, details=None):
        return self.decorated.addUnexpectedSuccess(test, details=details)

    def _get_failfast(self):
        return getattr(self.decorated, 'failfast', False)

    def _set_failfast(self, value):
        self.decorated.failfast = value
    failfast = property(_get_failfast, _set_failfast)

    def progress(self, offset, whence):
        return self.decorated.progress(offset, whence)

    def wasSuccessful(self):
        return self.decorated.wasSuccessful()

    @property
    def shouldStop(self):
        return self.decorated.shouldStop

    def stop(self):
        return self.decorated.stop()

    @property
    def testsRun(self):
        return self.decorated.testsRun

    def tags(self, new_tags, gone_tags):
        return self.decorated.tags(new_tags, gone_tags)

    def time(self, a_datetime):
        return self.decorated.time(a_datetime)