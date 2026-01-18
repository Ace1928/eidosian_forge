from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1FaceAnnotation(_messages.Message):
    """A face annotation object contains the results of face detection.

  Enums:
    AngerLikelihoodValueValuesEnum: Anger likelihood.
    BlurredLikelihoodValueValuesEnum: Blurred likelihood.
    HeadwearLikelihoodValueValuesEnum: Headwear likelihood.
    JoyLikelihoodValueValuesEnum: Joy likelihood.
    SorrowLikelihoodValueValuesEnum: Sorrow likelihood.
    SurpriseLikelihoodValueValuesEnum: Surprise likelihood.
    UnderExposedLikelihoodValueValuesEnum: Under-exposed likelihood.

  Fields:
    angerLikelihood: Anger likelihood.
    blurredLikelihood: Blurred likelihood.
    boundingPoly: The bounding polygon around the face. The coordinates of the
      bounding box are in the original image's scale. The bounding box is
      computed to "frame" the face in accordance with human expectations. It
      is based on the landmarker results. Note that one or more x and/or y
      coordinates may not be generated in the `BoundingPoly` (the polygon will
      be unbounded) if only a partial face appears in the image to be
      annotated.
    detectionConfidence: Detection confidence. Range [0, 1].
    fdBoundingPoly: The `fd_bounding_poly` bounding polygon is tighter than
      the `boundingPoly`, and encloses only the skin part of the face.
      Typically, it is used to eliminate the face from any image analysis that
      detects the "amount of skin" visible in an image. It is not based on the
      landmarker results, only on the initial face detection, hence the fd
      (face detection) prefix.
    headwearLikelihood: Headwear likelihood.
    joyLikelihood: Joy likelihood.
    landmarkingConfidence: Face landmarking confidence. Range [0, 1].
    landmarks: Detected face landmarks.
    panAngle: Yaw angle, which indicates the leftward/rightward angle that the
      face is pointing relative to the vertical plane perpendicular to the
      image. Range [-180,180].
    rollAngle: Roll angle, which indicates the amount of clockwise/anti-
      clockwise rotation of the face relative to the image vertical about the
      axis perpendicular to the face. Range [-180,180].
    sorrowLikelihood: Sorrow likelihood.
    surpriseLikelihood: Surprise likelihood.
    tiltAngle: Pitch angle, which indicates the upwards/downwards angle that
      the face is pointing relative to the image's horizontal plane. Range
      [-180,180].
    underExposedLikelihood: Under-exposed likelihood.
  """

    class AngerLikelihoodValueValuesEnum(_messages.Enum):
        """Anger likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class BlurredLikelihoodValueValuesEnum(_messages.Enum):
        """Blurred likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class HeadwearLikelihoodValueValuesEnum(_messages.Enum):
        """Headwear likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class JoyLikelihoodValueValuesEnum(_messages.Enum):
        """Joy likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class SorrowLikelihoodValueValuesEnum(_messages.Enum):
        """Sorrow likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class SurpriseLikelihoodValueValuesEnum(_messages.Enum):
        """Surprise likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class UnderExposedLikelihoodValueValuesEnum(_messages.Enum):
        """Under-exposed likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5
    angerLikelihood = _messages.EnumField('AngerLikelihoodValueValuesEnum', 1)
    blurredLikelihood = _messages.EnumField('BlurredLikelihoodValueValuesEnum', 2)
    boundingPoly = _messages.MessageField('GoogleCloudVisionV1p1beta1BoundingPoly', 3)
    detectionConfidence = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    fdBoundingPoly = _messages.MessageField('GoogleCloudVisionV1p1beta1BoundingPoly', 5)
    headwearLikelihood = _messages.EnumField('HeadwearLikelihoodValueValuesEnum', 6)
    joyLikelihood = _messages.EnumField('JoyLikelihoodValueValuesEnum', 7)
    landmarkingConfidence = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    landmarks = _messages.MessageField('GoogleCloudVisionV1p1beta1FaceAnnotationLandmark', 9, repeated=True)
    panAngle = _messages.FloatField(10, variant=_messages.Variant.FLOAT)
    rollAngle = _messages.FloatField(11, variant=_messages.Variant.FLOAT)
    sorrowLikelihood = _messages.EnumField('SorrowLikelihoodValueValuesEnum', 12)
    surpriseLikelihood = _messages.EnumField('SurpriseLikelihoodValueValuesEnum', 13)
    tiltAngle = _messages.FloatField(14, variant=_messages.Variant.FLOAT)
    underExposedLikelihood = _messages.EnumField('UnderExposedLikelihoodValueValuesEnum', 15)