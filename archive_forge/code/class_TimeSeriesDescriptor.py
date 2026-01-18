from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesDescriptor(_messages.Message):
    """A descriptor for the labels and points in a time series.

  Fields:
    labelDescriptors: Descriptors for the labels.
    pointDescriptors: Descriptors for the point data value columns.
  """
    labelDescriptors = _messages.MessageField('LabelDescriptor', 1, repeated=True)
    pointDescriptors = _messages.MessageField('ValueDescriptor', 2, repeated=True)