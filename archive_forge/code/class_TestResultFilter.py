import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TestResultFilter(TestResultDecorator):
    """A pyunit TestResult interface implementation which filters tests.

    Tests that pass the filter are handed on to another TestResult instance
    for further processing/reporting. To obtain the filtered results,
    the other instance must be interrogated.

    :ivar result: The result that tests are passed to after filtering.
    :ivar filter_predicate: The callback run to decide whether to pass
        a result.
    """

    def __init__(self, result, filter_error=False, filter_failure=False, filter_success=True, filter_skip=False, filter_xfail=False, filter_predicate=None, fixup_expected_failures=None, rename=None):
        """Create a FilterResult object filtering to result.

        :param filter_error: Filter out errors.
        :param filter_failure: Filter out failures.
        :param filter_success: Filter out successful tests.
        :param filter_skip: Filter out skipped tests.
        :param filter_xfail: Filter out expected failure tests.
        :param filter_predicate: A callable taking (test, outcome, err,
            details, tags) and returning True if the result should be passed
            through.  err and details may be none if no error or extra
            metadata is available. outcome is the name of the outcome such
            as 'success' or 'failure'. tags is new in 0.0.8; 0.0.7 filters
            are still supported but should be updated to accept the tags
            parameter for efficiency.
        :param fixup_expected_failures: Set of test ids to consider known
            failing.
        :param rename: Optional function to rename test ids
        """
        predicates = []
        if filter_error:
            predicates.append(lambda t, outcome, e, d, tags: outcome != 'error')
        if filter_failure:
            predicates.append(lambda t, outcome, e, d, tags: outcome != 'failure')
        if filter_success:
            predicates.append(lambda t, outcome, e, d, tags: outcome != 'success')
        if filter_skip:
            predicates.append(lambda t, outcome, e, d, tags: outcome != 'skip')
        if filter_xfail:
            predicates.append(lambda t, outcome, e, d, tags: outcome != 'expectedfailure')
        if filter_predicate is not None:

            def compat(test, outcome, error, details, tags):
                try:
                    return filter_predicate(test, outcome, error, details, tags)
                except TypeError:
                    return filter_predicate(test, outcome, error, details)
            predicates.append(compat)
        predicate = and_predicates(predicates)
        super().__init__(_PredicateFilter(result, predicate))
        if fixup_expected_failures is None:
            self._fixup_expected_failures = frozenset()
        else:
            self._fixup_expected_failures = fixup_expected_failures
        self._rename_fn = rename

    def addError(self, test, err=None, details=None):
        test = self._apply_renames(test)
        if self._failure_expected(test):
            self.addExpectedFailure(test, err=err, details=details)
        else:
            super().addError(test, err=err, details=details)

    def addFailure(self, test, err=None, details=None):
        test = self._apply_renames(test)
        if self._failure_expected(test):
            self.addExpectedFailure(test, err=err, details=details)
        else:
            super().addFailure(test, err=err, details=details)

    def addSuccess(self, test, details=None):
        test = self._apply_renames(test)
        if self._failure_expected(test):
            self.addUnexpectedSuccess(test, details=details)
        else:
            super().addSuccess(test, details=details)

    def _failure_expected(self, test):
        return test.id() in self._fixup_expected_failures

    def _apply_renames(self, test):
        if self._rename_fn is None:
            return test
        new_id = self._rename_fn(test.id())
        setattr(test, 'id', lambda: new_id)
        return test