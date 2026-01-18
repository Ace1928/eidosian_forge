from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntelligenceCloudAutomlXpsMetricEntry(_messages.Message):
    """A IntelligenceCloudAutomlXpsMetricEntry object.

  Fields:
    argentumMetricId: For billing metrics that are using legacy sku's, set the
      legacy billing metric id here. This will be sent to Chemist as the
      "cloudbilling.googleapis.com/argentum_metric_id" label. Otherwise leave
      empty.
    doubleValue: A double value.
    int64Value: A signed 64-bit integer value.
    metricName: The metric name defined in the service configuration.
    systemLabels: Billing system labels for this (metric, value) pair.
  """
    argentumMetricId = _messages.StringField(1)
    doubleValue = _messages.FloatField(2)
    int64Value = _messages.IntegerField(3)
    metricName = _messages.StringField(4)
    systemLabels = _messages.MessageField('IntelligenceCloudAutomlXpsMetricEntryLabel', 5, repeated=True)