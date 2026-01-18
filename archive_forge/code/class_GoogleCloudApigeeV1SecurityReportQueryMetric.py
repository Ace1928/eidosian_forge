from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityReportQueryMetric(_messages.Message):
    """Metric of the Query

  Fields:
    aggregationFunction: Aggregation function: avg, min, max, or sum.
    alias: Alias for the metric. Alias will be used to replace metric name in
      query results.
    name: Required. Metric name.
    operator: One of `+`, `-`, `/`, `%`, `*`.
    value: Operand value should be provided when operator is set.
  """
    aggregationFunction = _messages.StringField(1)
    alias = _messages.StringField(2)
    name = _messages.StringField(3)
    operator = _messages.StringField(4)
    value = _messages.StringField(5)