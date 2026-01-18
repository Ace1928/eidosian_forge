from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2Vertex(_messages.Message):
    """A vertex represents a 2D point in the image. NOTE: the vertex
  coordinates are in the same scale as the original image.

  Fields:
    x: X coordinate.
    y: Y coordinate (starts from the top of the image).
  """
    x = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    y = _messages.IntegerField(2, variant=_messages.Variant.INT32)