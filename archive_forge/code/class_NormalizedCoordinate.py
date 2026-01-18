from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NormalizedCoordinate(_messages.Message):
    """2D normalized coordinates. Default: `{0.0, 0.0}`

  Fields:
    x: Normalized x coordinate.
    y: Normalized y coordinate.
  """
    x = _messages.FloatField(1)
    y = _messages.FloatField(2)