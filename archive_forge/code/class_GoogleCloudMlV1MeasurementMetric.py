from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1MeasurementMetric(_messages.Message):
    """A message representing a metric in the measurement.

  Fields:
    metric: Required. Metric name.
    value: Required. The value for this metric.
  """
    metric = _messages.StringField(1)
    value = _messages.FloatField(2)