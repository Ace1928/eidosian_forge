from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlPlanMetric(_messages.Message):
    """Metrics related to SQL execution.

  Fields:
    accumulatorId: A string attribute.
    metricType: A string attribute.
    name: A string attribute.
  """
    accumulatorId = _messages.IntegerField(1)
    metricType = _messages.StringField(2)
    name = _messages.StringField(3)