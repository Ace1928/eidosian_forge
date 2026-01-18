from collections import namedtuple
from testtools.tags import TagContext
class Python27TestResult(Python26TestResult):
    """A precisely python 2.7 like test result, that logs."""

    def __init__(self, event_log=None):
        super().__init__(event_log)
        self.failfast = False

    def addError(self, test, err):
        super().addError(test, err)
        if self.failfast:
            self.stop()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.failfast:
            self.stop()

    def addExpectedFailure(self, test, err):
        self._events.append(('addExpectedFailure', test, err))

    def addSkip(self, test, reason):
        self._events.append(('addSkip', test, reason))

    def addUnexpectedSuccess(self, test):
        self._events.append(('addUnexpectedSuccess', test))
        if self.failfast:
            self.stop()

    def startTestRun(self):
        self._events.append(('startTestRun',))

    def stopTestRun(self):
        self._events.append(('stopTestRun',))