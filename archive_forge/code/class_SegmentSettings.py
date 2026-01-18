from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SegmentSettings(_messages.Message):
    """Segment settings for `ts`, `fmp4` and `vtt`.

  Fields:
    individualSegments: Required. Create an individual segment file. The
      default is `false`.
    segmentDuration: Duration of the segments in seconds. The default is
      `6.0s`. Note that `segmentDuration` must be greater than or equal to
      [`gopDuration`](#videostream), and `segmentDuration` must be divisible
      by [`gopDuration`](#videostream).
  """
    individualSegments = _messages.BooleanField(1)
    segmentDuration = _messages.StringField(2)