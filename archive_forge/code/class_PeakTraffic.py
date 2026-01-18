from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeakTraffic(_messages.Message):
    """Expected peak traffic between two coordinates.

  Fields:
    peakTrafficGbps: Gigabits per second.
  """
    peakTrafficGbps = _messages.FloatField(1)