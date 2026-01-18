from unittest import TestResult
import testresources
from testresources.tests import TestUtil
class ResultWithResourceExtensions(TestResult):
    """A test fake which has resource extensions."""

    def __init__(self):
        TestResult.__init__(self)
        self._calls = []

    def startCleanResource(self, resource):
        self._calls.append(('clean', 'start', resource))

    def stopCleanResource(self, resource):
        self._calls.append(('clean', 'stop', resource))

    def startMakeResource(self, resource):
        self._calls.append(('make', 'start', resource))

    def stopMakeResource(self, resource):
        self._calls.append(('make', 'stop', resource))

    def startResetResource(self, resource):
        self._calls.append(('reset', 'start', resource))

    def stopResetResource(self, resource):
        self._calls.append(('reset', 'stop', resource))