from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoteServiceTypeValueValuesEnum(_messages.Enum):
    """Output only. The remote service type for remote model.

    Values:
      REMOTE_SERVICE_TYPE_UNSPECIFIED: Unspecified remote service type.
      CLOUD_AI_TRANSLATE_V3: V3 Cloud AI Translation API. See more details at
        [Cloud Translation API]
        (https://cloud.google.com/translate/docs/reference/rest).
      CLOUD_AI_VISION_V1: V1 Cloud AI Vision API See more details at [Cloud
        Vision API] (https://cloud.google.com/vision/docs/reference/rest).
      CLOUD_AI_NATURAL_LANGUAGE_V1: V1 Cloud AI Natural Language API. See more
        details at [REST Resource:
        documents](https://cloud.google.com/natural-
        language/docs/reference/rest/v1/documents).
      CLOUD_AI_SPEECH_TO_TEXT_V2: V2 Speech-to-Text API. See more details at
        [Google Cloud Speech-to-Text V2 API](https://cloud.google.com/speech-
        to-text/v2/docs)
    """
    REMOTE_SERVICE_TYPE_UNSPECIFIED = 0
    CLOUD_AI_TRANSLATE_V3 = 1
    CLOUD_AI_VISION_V1 = 2
    CLOUD_AI_NATURAL_LANGUAGE_V1 = 3
    CLOUD_AI_SPEECH_TO_TEXT_V2 = 4