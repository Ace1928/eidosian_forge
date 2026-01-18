from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1VideoAnnotationProgress(_messages.Message):
    """Annotation progress for a single video.

  Enums:
    FeatureValueValuesEnum: Specifies which feature is being tracked if the
      request contains more than one feature.

  Fields:
    feature: Specifies which feature is being tracked if the request contains
      more than one feature.
    inputUri: Video file location in [Cloud
      Storage](https://cloud.google.com/storage/).
    progressPercent: Approximate percentage processed thus far. Guaranteed to
      be 100 when fully processed.
    segment: Specifies which segment is being tracked if the request contains
      more than one segment.
    startTime: Time when the request was received.
    updateTime: Time of the most recent update.
  """

    class FeatureValueValuesEnum(_messages.Enum):
        """Specifies which feature is being tracked if the request contains more
    than one feature.

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
    feature = _messages.EnumField('FeatureValueValuesEnum', 1)
    inputUri = _messages.StringField(2)
    progressPercent = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1VideoSegment', 4)
    startTime = _messages.StringField(5)
    updateTime = _messages.StringField(6)