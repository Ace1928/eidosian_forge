from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetEnvironmentOutcomeDetails(self, environment):
    """Turn test outcome counts and details into something human readable."""
    outcome = environment.environmentResult.outcome
    summary_enum = self._messages.Outcome.SummaryValueValuesEnum
    test_suite_overviews = environment.environmentResult.testSuiteOverviews
    if outcome.summary == summary_enum.success:
        details = _GetSuccessCountDetails(test_suite_overviews)
        if outcome.successDetail and outcome.successDetail.otherNativeCrash:
            return '{d} ({c})'.format(d=details, c=_NATIVE_CRASH)
        else:
            return details
    elif outcome.summary == summary_enum.failure or outcome.summary == summary_enum.flaky:
        if outcome.failureDetail:
            return _GetFailureDetail(outcome, test_suite_overviews)
        return _GetFailureOrFlakyCountDetails(test_suite_overviews)
    elif outcome.summary == summary_enum.inconclusive:
        return _GetInconclusiveDetail(outcome)
    elif outcome.summary == summary_enum.skipped:
        return _GetSkippedDetail(outcome)
    else:
        return 'Unknown outcome'