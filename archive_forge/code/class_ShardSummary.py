from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShardSummary(_messages.Message):
    """Result summary for a shard in an environment.

  Fields:
    runs: Summaries of the steps belonging to the shard. With
      flaky_test_attempts enabled from TestExecutionService, more than one run
      (Step) can present. And the runs will be sorted by multistep_number.
    shardResult: Merged result of the shard.
  """
    runs = _messages.MessageField('StepSummary', 1, repeated=True)
    shardResult = _messages.MessageField('MergedResult', 2)