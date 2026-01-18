from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Segment(_messages.Message):
    """Represents a segment of a video asset that matches a search query.

  Fields:
    endOffset: Segment end offet timestamp.
    startOffset: Segment start offset timestamp.
  """
    endOffset = _messages.StringField(1)
    startOffset = _messages.StringField(2)