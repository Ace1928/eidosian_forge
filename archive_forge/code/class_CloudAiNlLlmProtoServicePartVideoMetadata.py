from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePartVideoMetadata(_messages.Message):
    """Metadata describes the input video content.

  Fields:
    endOffset: The end offset of the video.
    startOffset: The start offset of the video.
  """
    endOffset = _messages.StringField(1)
    startOffset = _messages.StringField(2)