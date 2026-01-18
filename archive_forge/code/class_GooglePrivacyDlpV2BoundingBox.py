from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BoundingBox(_messages.Message):
    """Bounding box encompassing detected text within an image.

  Fields:
    height: Height of the bounding box in pixels.
    left: Left coordinate of the bounding box. (0,0) is upper left.
    top: Top coordinate of the bounding box. (0,0) is upper left.
    width: Width of the bounding box in pixels.
  """
    height = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    left = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    top = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    width = _messages.IntegerField(4, variant=_messages.Variant.INT32)