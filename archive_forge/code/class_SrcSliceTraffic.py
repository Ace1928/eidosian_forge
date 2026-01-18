from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SrcSliceTraffic(_messages.Message):
    """All the non-zero edges of traffic leaving a src `slice_coord` directed
  towards dst slices.

  Fields:
    dstTraffic: List of traffic edges directed towards dst slices.
    sliceCoord: Src slice coordinate.
  """
    dstTraffic = _messages.MessageField('DstSliceTraffic', 1, repeated=True)
    sliceCoord = _messages.IntegerField(2, variant=_messages.Variant.INT32)