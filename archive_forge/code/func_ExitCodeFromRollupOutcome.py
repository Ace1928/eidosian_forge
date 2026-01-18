from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def ExitCodeFromRollupOutcome(outcome, summary_enum):
    """Map a test roll-up outcome into the appropriate gcloud test exit_code.

  Args:
    outcome: a toolresults_v1.Outcome message.
    summary_enum: a toolresults.Outcome.SummaryValueValuesEnum reference.

  Returns:
    The exit_code which corresponds to the test execution's rolled-up outcome.

  Raises:
    TestOutcomeError: If Tool Results service returns an invalid outcome value.
  """
    if not outcome or not outcome.summary:
        log.warning('Tool Results service did not provide a roll-up test outcome.')
        return INCONCLUSIVE
    if outcome.summary == summary_enum.success or outcome.summary == summary_enum.flaky:
        return ROLLUP_SUCCESS
    if outcome.summary == summary_enum.failure:
        return ROLLUP_FAILURE
    if outcome.summary == summary_enum.skipped:
        return UNSUPPORTED_ENV
    if outcome.summary == summary_enum.inconclusive:
        return INCONCLUSIVE
    raise TestOutcomeError("Unknown test outcome summary value '{0}'".format(outcome.summary))