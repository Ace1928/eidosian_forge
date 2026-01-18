from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PointData(_messages.Message):
    """A point's value columns and time interval. Each point has one or more
  point values corresponding to the entries in point_descriptors field in the
  TimeSeriesDescriptor associated with this object.

  Fields:
    timeInterval: The time interval associated with the point.
    values: The values that make up the point.
  """
    timeInterval = _messages.MessageField('TimeInterval', 1)
    values = _messages.MessageField('TypedValue', 2, repeated=True)