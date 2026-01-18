import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class _PredicateFilter(TestResultDecorator, TagsMixin):

    def __init__(self, result, predicate):
        super().__init__(result)
        self._clear_tags()
        self.decorated = TimeCollapsingDecorator(TagCollapsingDecorator(self.decorated))
        self._predicate = predicate
        self._current_test = None
        self._current_test_filtered = None
        self._buffered_calls = []

    def filter_predicate(self, test, outcome, error, details):
        return self._predicate(test, outcome, error, details, self._get_active_tags())

    def addError(self, test, err=None, details=None):
        if self.filter_predicate(test, 'error', err, details):
            self._buffered_calls.append(('addError', [test, err], {'details': details}))
        else:
            self._filtered()

    def addFailure(self, test, err=None, details=None):
        if self.filter_predicate(test, 'failure', err, details):
            self._buffered_calls.append(('addFailure', [test, err], {'details': details}))
        else:
            self._filtered()

    def addSkip(self, test, reason=None, details=None):
        if self.filter_predicate(test, 'skip', reason, details):
            self._buffered_calls.append(('addSkip', [test, reason], {'details': details}))
        else:
            self._filtered()

    def addExpectedFailure(self, test, err=None, details=None):
        if self.filter_predicate(test, 'expectedfailure', err, details):
            self._buffered_calls.append(('addExpectedFailure', [test, err], {'details': details}))
        else:
            self._filtered()

    def addUnexpectedSuccess(self, test, details=None):
        self._buffered_calls.append(('addUnexpectedSuccess', [test], {'details': details}))

    def addSuccess(self, test, details=None):
        if self.filter_predicate(test, 'success', None, details):
            self._buffered_calls.append(('addSuccess', [test], {'details': details}))
        else:
            self._filtered()

    def _filtered(self):
        self._current_test_filtered = True

    def startTest(self, test):
        """Start a test.

        Not directly passed to the client, but used for handling of tags
        correctly.
        """
        TagsMixin.startTest(self, test)
        self._current_test = test
        self._current_test_filtered = False
        self._buffered_calls.append(('startTest', [test], {}))

    def stopTest(self, test):
        """Stop a test.

        Not directly passed to the client, but used for handling of tags
        correctly.
        """
        if not self._current_test_filtered:
            for method, args, kwargs in self._buffered_calls:
                getattr(self.decorated, method)(*args, **kwargs)
            self.decorated.stopTest(test)
        self._current_test = None
        self._current_test_filtered = None
        self._buffered_calls = []
        TagsMixin.stopTest(self, test)

    def tags(self, new_tags, gone_tags):
        TagsMixin.tags(self, new_tags, gone_tags)
        if self._current_test is not None:
            self._buffered_calls.append(('tags', [new_tags, gone_tags], {}))
        else:
            return super().tags(new_tags, gone_tags)

    def time(self, a_time):
        return self.decorated.time(a_time)

    def id_to_orig_id(self, id):
        if id.startswith('subunit.RemotedTestCase.'):
            return id[len('subunit.RemotedTestCase.'):]
        return id