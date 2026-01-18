from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginalMediaTypeValueValuesEnum(_messages.Enum):
    """The original media the speech was recorded on.

    Values:
      ORIGINAL_MEDIA_TYPE_UNSPECIFIED: Unknown original media type.
      AUDIO: The speech data is an audio recording.
      VIDEO: The speech data originally recorded on a video.
    """
    ORIGINAL_MEDIA_TYPE_UNSPECIFIED = 0
    AUDIO = 1
    VIDEO = 2