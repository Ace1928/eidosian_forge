from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MeasurementMetric(_messages.Message):
    """A message representing a metric in the measurement.

  Fields:
    metricId: Output only. The ID of the Metric. The Metric should be defined
      in StudySpec's Metrics.
    value: Output only. The value for this metric.
  """
    metricId = _messages.StringField(1)
    value = _messages.FloatField(2)