from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndividualOutcome(_messages.Message):
    """Step Id and outcome of each individual step that was run as a group with
  other steps with the same configuration.

  Enums:
    OutcomeSummaryValueValuesEnum:

  Fields:
    multistepNumber: Unique int given to each step. Ranges from 0(inclusive)
      to total number of steps(exclusive). The primary step is 0.
    outcomeSummary: A OutcomeSummaryValueValuesEnum attribute.
    runDuration: How long it took for this step to run.
    stepId: A string attribute.
  """

    class OutcomeSummaryValueValuesEnum(_messages.Enum):
        """OutcomeSummaryValueValuesEnum enum type.

    Values:
      unset: Do not use. For proto versioning only.
      success: The test matrix run was successful, for instance: - All the
        test cases passed. - Robo did not detect a crash of the application
        under test.
      failure: A run failed, for instance: - One or more test case failed. - A
        test timed out. - The application under test crashed.
      inconclusive: Something unexpected happened. The run should still be
        considered unsuccessful but this is likely a transient problem and re-
        running the test might be successful.
      skipped: All tests were skipped, for instance: - All device
        configurations were incompatible.
      flaky: A group of steps that were run with the same configuration had
        both failure and success outcomes.
    """
        unset = 0
        success = 1
        failure = 2
        inconclusive = 3
        skipped = 4
        flaky = 5
    multistepNumber = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    outcomeSummary = _messages.EnumField('OutcomeSummaryValueValuesEnum', 2)
    runDuration = _messages.MessageField('Duration', 3)
    stepId = _messages.StringField(4)