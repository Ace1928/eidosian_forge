from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeaturesValueListEntryValuesEnum(_messages.Enum):
    """FeaturesValueListEntryValuesEnum enum type.

    Values:
      FEATURE_UNSPECIFIED: Unspecified.
      LABEL_DETECTION: Label detection. Detect objects, such as dog or flower.
      SHOT_CHANGE_DETECTION: Shot change detection.
      EXPLICIT_CONTENT_DETECTION: Explicit content detection.
      FACE_DETECTION: Human face detection.
      SPEECH_TRANSCRIPTION: Speech transcription.
      TEXT_DETECTION: OCR text detection and tracking.
      OBJECT_TRACKING: Object detection and tracking.
      LOGO_RECOGNITION: Logo detection, tracking, and recognition.
      PERSON_DETECTION: Person detection.
    """
    FEATURE_UNSPECIFIED = 0
    LABEL_DETECTION = 1
    SHOT_CHANGE_DETECTION = 2
    EXPLICIT_CONTENT_DETECTION = 3
    FACE_DETECTION = 4
    SPEECH_TRANSCRIPTION = 5
    TEXT_DETECTION = 6
    OBJECT_TRACKING = 7
    LOGO_RECOGNITION = 8
    PERSON_DETECTION = 9