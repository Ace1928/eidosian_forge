from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageMediaContent(_messages.Message):
    """The media content card for Actions on Google.

  Enums:
    MediaTypeValueValuesEnum: Optional. What type of media is the content (ie
      "audio").

  Fields:
    mediaObjects: Required. List of media objects.
    mediaType: Optional. What type of media is the content (ie "audio").
  """

    class MediaTypeValueValuesEnum(_messages.Enum):
        """Optional. What type of media is the content (ie "audio").

    Values:
      RESPONSE_MEDIA_TYPE_UNSPECIFIED: Unspecified.
      AUDIO: Response media type is audio.
    """
        RESPONSE_MEDIA_TYPE_UNSPECIFIED = 0
        AUDIO = 1
    mediaObjects = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageMediaContentResponseMediaObject', 1, repeated=True)
    mediaType = _messages.EnumField('MediaTypeValueValuesEnum', 2)