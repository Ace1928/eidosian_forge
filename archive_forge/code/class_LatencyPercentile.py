from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatencyPercentile(_messages.Message):
    """Latency percentile rank and value.

  Fields:
    latencyMicros: percent-th percentile of latency observed, in microseconds.
      Fraction of percent/100 of samples have latency lower or equal to the
      value of this field.
    percent: Percentage of samples this data point applies to.
  """
    latencyMicros = _messages.IntegerField(1)
    percent = _messages.IntegerField(2, variant=_messages.Variant.INT32)