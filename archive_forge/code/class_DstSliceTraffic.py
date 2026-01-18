from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DstSliceTraffic(_messages.Message):
    """A single edge of traffic directed towards a dst `slice_coord`.

  Fields:
    sliceCoord: Dst slice coordinate.
    traffic: Traffic directed towards this slice.
  """
    sliceCoord = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    traffic = _messages.MessageField('Traffic', 2)