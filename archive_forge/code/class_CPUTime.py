from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CPUTime(_messages.Message):
    """Modeled after information exposed by /proc/stat.

  Fields:
    rate: Average CPU utilization rate (% non-idle cpu / second) since
      previous sample.
    timestamp: Timestamp of the measurement.
    totalMs: Total active CPU time across all cores (ie., non-idle) in
      milliseconds since start-up.
  """
    rate = _messages.FloatField(1)
    timestamp = _messages.StringField(2)
    totalMs = _messages.IntegerField(3, variant=_messages.Variant.UINT64)