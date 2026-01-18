from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Segment(_messages.Message):
    """Segment of the content.

  Fields:
    endIndex: Output only. End index in the given Part, measured in bytes.
      Offset from the start of the Part, exclusive, starting at zero.
    partIndex: Output only. The index of a Part object within its parent
      Content object.
    startIndex: Output only. Start index in the given Part, measured in bytes.
      Offset from the start of the Part, inclusive, starting at zero.
  """
    endIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    partIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    startIndex = _messages.IntegerField(3, variant=_messages.Variant.INT32)