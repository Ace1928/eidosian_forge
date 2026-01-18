from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OptimizedStatsResponse(_messages.Message):
    """Encapsulates a response format for JavaScript Optimized Scenario.

  Fields:
    TimeUnit: List of time unit values. Time unit refers to an epoch timestamp
      value.
    metaData: Metadata information about the query executed.
    resultTruncated: Boolean flag that indicates whether the results were
      truncated based on the limit parameter.
    stats: `stats` results.
  """
    TimeUnit = _messages.IntegerField(1, repeated=True)
    metaData = _messages.MessageField('GoogleCloudApigeeV1Metadata', 2)
    resultTruncated = _messages.BooleanField(3)
    stats = _messages.MessageField('GoogleCloudApigeeV1OptimizedStatsNode', 4)