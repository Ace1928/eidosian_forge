import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class MultiTestResult(TestResult):
    """A test result that dispatches to many test results."""

    def __init__(self, *results):
        self._results = list(map(ExtendedToOriginalDecorator, results))
        super().__init__()

    def __repr__(self):
        return '<{} ({})>'.format(self.__class__.__name__, ', '.join(map(repr, self._results)))

    def _dispatch(self, message, *args, **kwargs):
        return tuple((getattr(result, message)(*args, **kwargs) for result in self._results))

    def _get_failfast(self):
        return getattr(self._results[0], 'failfast', False)

    def _set_failfast(self, value):
        self._dispatch('__setattr__', 'failfast', value)
    failfast = property(_get_failfast, _set_failfast)

    def _get_shouldStop(self):
        return any(self._dispatch('__getattr__', 'shouldStop'))

    def _set_shouldStop(self, value):
        pass
    shouldStop = property(_get_shouldStop, _set_shouldStop)

    def startTest(self, test):
        super().startTest(test)
        return self._dispatch('startTest', test)

    def stop(self):
        return self._dispatch('stop')

    def stopTest(self, test):
        super().stopTest(test)
        return self._dispatch('stopTest', test)

    def addError(self, test, error=None, details=None):
        return self._dispatch('addError', test, error, details=details)

    def addExpectedFailure(self, test, err=None, details=None):
        return self._dispatch('addExpectedFailure', test, err, details=details)

    def addFailure(self, test, err=None, details=None):
        return self._dispatch('addFailure', test, err, details=details)

    def addSkip(self, test, reason=None, details=None):
        return self._dispatch('addSkip', test, reason, details=details)

    def addSuccess(self, test, details=None):
        return self._dispatch('addSuccess', test, details=details)

    def addUnexpectedSuccess(self, test, details=None):
        return self._dispatch('addUnexpectedSuccess', test, details=details)

    def startTestRun(self):
        super().startTestRun()
        return self._dispatch('startTestRun')

    def stopTestRun(self):
        return self._dispatch('stopTestRun')

    def tags(self, new_tags, gone_tags):
        super().tags(new_tags, gone_tags)
        return self._dispatch('tags', new_tags, gone_tags)

    def time(self, a_datetime):
        return self._dispatch('time', a_datetime)

    def done(self):
        return self._dispatch('done')

    def wasSuccessful(self):
        """Was this result successful?

        Only returns True if every constituent result was successful.
        """
        return all(self._dispatch('wasSuccessful'))