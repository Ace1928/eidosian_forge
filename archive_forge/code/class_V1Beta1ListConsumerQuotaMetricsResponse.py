from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ListConsumerQuotaMetricsResponse(_messages.Message):
    """Response message for ListConsumerQuotaMetrics.

  Fields:
    metrics: Quota settings for the consumer, organized by quota metric.
    nextPageToken: Token identifying which result to start with; returned by a
      previous list call.
  """
    metrics = _messages.MessageField('V1Beta1ConsumerQuotaMetric', 1, repeated=True)
    nextPageToken = _messages.StringField(2)