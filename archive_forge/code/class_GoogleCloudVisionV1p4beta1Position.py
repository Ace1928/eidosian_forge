from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1Position(_messages.Message):
    """A 3D position in the image, used primarily for Face detection landmarks.
  A valid Position must have both x and y coordinates. The position
  coordinates are in the same scale as the original image.

  Fields:
    x: X coordinate.
    y: Y coordinate.
    z: Z coordinate (or depth).
  """
    x = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    y = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    z = _messages.FloatField(3, variant=_messages.Variant.FLOAT)