from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorPeakMetricsDistributions(_messages.Message):
    """A ExecutorPeakMetricsDistributions object.

  Fields:
    executorMetrics: A ExecutorMetrics attribute.
    quantiles: A number attribute.
  """
    executorMetrics = _messages.MessageField('ExecutorMetrics', 1, repeated=True)
    quantiles = _messages.FloatField(2, repeated=True)