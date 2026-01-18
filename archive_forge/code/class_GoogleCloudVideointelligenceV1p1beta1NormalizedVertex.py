from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1NormalizedVertex(_messages.Message):
    """A vertex represents a 2D point in the image. NOTE: the normalized vertex
  coordinates are relative to the original image and range from 0 to 1.

  Fields:
    x: X coordinate.
    y: Y coordinate.
  """
    x = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    y = _messages.FloatField(2, variant=_messages.Variant.FLOAT)