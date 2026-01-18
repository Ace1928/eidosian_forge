from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesData(_messages.Message):
    """Represents the values of a time series associated with a
  TimeSeriesDescriptor.

  Fields:
    labelValues: The values of the labels in the time series identifier, given
      in the same order as the label_descriptors field of the
      TimeSeriesDescriptor associated with this object. Each value must have a
      value of the type given in the corresponding entry of label_descriptors.
    pointData: The points in the time series.
  """
    labelValues = _messages.MessageField('LabelValue', 1, repeated=True)
    pointData = _messages.MessageField('PointData', 2, repeated=True)