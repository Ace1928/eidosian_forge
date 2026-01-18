from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionVideoActionRecognitionPredictionResult(_messages.Message):
    """Prediction output format for Video Action Recognition.

  Fields:
    confidence: The Model's confidence in correction of this prediction,
      higher value means higher confidence.
    displayName: The display name of the AnnotationSpec that had been
      identified.
    id: The resource ID of the AnnotationSpec that had been identified.
    timeSegmentEnd: The end, exclusive, of the video's time segment in which
      the AnnotationSpec has been identified. Expressed as a number of seconds
      as measured from the start of the video, with fractions up to a
      microsecond precision, and with "s" appended at the end.
    timeSegmentStart: The beginning, inclusive, of the video's time segment in
      which the AnnotationSpec has been identified. Expressed as a number of
      seconds as measured from the start of the video, with fractions up to a
      microsecond precision, and with "s" appended at the end.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    displayName = _messages.StringField(2)
    id = _messages.StringField(3)
    timeSegmentEnd = _messages.StringField(4)
    timeSegmentStart = _messages.StringField(5)