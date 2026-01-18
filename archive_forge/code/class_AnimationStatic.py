from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnimationStatic(_messages.Message):
    """Display static overlay object.

  Fields:
    startTimeOffset: The time to start displaying the overlay object, in
      seconds. Default: 0
    xy: Normalized coordinates based on output video resolution. Valid values:
      `0.0`\\u2013`1.0`. `xy` is the upper-left coordinate of the overlay
      object. For example, use the x and y coordinates {0,0} to position the
      top-left corner of the overlay animation in the top-left corner of the
      output video.
  """
    startTimeOffset = _messages.StringField(1)
    xy = _messages.MessageField('NormalizedCoordinate', 2)