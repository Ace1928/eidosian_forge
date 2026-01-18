from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def CreateMatrixOutcomeSummaryUsingSteps(self):
    """Fetches test results and creates a test outcome summary.

    Lists all the steps in an execution and creates a high-level outcome summary
    for each step (pass/fail/inconclusive). Each step represents a test run on
    a single device (e.g. running the tests on a Nexus 5 in portrait mode using
    the en locale and API level 18).

    Returns:
      A list of TestOutcome objects.

    Raises:
      HttpException if the ToolResults service reports a back-end error.
    """
    outcomes = []
    steps = self._ListAllSteps()
    if not steps:
        log.warning('No test results found, something went wrong. Try re-running the tests.')
        return outcomes
    for step in steps:
        dimension_value = step.dimensionValue
        axis_value = self._GetAxisValue(dimension_value)
        if not step.outcome:
            log.warning('Step for [{0}] had no outcome value.'.format(axis_value))
        else:
            details = self._GetStepOutcomeDetails(step)
            self._LogWarnings(details, axis_value)
            outcome_summary = step.outcome.summary
            outcome_str = self._GetOutcomeSummaryDisplayName(outcome_summary)
            outcomes.append(TestOutcome(outcome=outcome_str, axis_value=axis_value, test_details=details))
    return sorted(outcomes, key=_TestOutcomeSortKey)