from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CpuUtilization(_messages.Message):
    """Target scaling by CPU usage.

  Fields:
    aggregationWindowLength: Period of time over which CPU utilization is
      calculated.
    targetUtilization: Target CPU utilization ratio to maintain when scaling.
      Must be between 0 and 1.
  """
    aggregationWindowLength = _messages.StringField(1)
    targetUtilization = _messages.FloatField(2)