from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SmartReplyMetrics(_messages.Message):
    """The evaluation metrics for smart reply model.

  Fields:
    allowlistCoverage: Percentage of target participant messages in the
      evaluation dataset for which similar messages have appeared at least
      once in the allowlist. Should be [0, 1].
    conversationCount: Total number of conversations used to generate this
      metric.
    topNMetrics: Metrics of top n smart replies, sorted by TopNMetric.n.
  """
    allowlistCoverage = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    conversationCount = _messages.IntegerField(2)
    topNMetrics = _messages.MessageField('GoogleCloudDialogflowV2SmartReplyMetricsTopNMetrics', 3, repeated=True)