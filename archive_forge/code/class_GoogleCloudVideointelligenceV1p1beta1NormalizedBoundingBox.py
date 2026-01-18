from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1NormalizedBoundingBox(_messages.Message):
    """Normalized bounding box. The normalized vertex coordinates are relative
  to the original image. Range: [0, 1].

  Fields:
    bottom: Bottom Y coordinate.
    left: Left X coordinate.
    right: Right X coordinate.
    top: Top Y coordinate.
  """
    bottom = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    left = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    right = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    top = _messages.FloatField(4, variant=_messages.Variant.FLOAT)