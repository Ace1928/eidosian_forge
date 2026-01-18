from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1FaceAnnotationLandmark(_messages.Message):
    """A face-specific landmark (for example, a face feature).

  Enums:
    TypeValueValuesEnum: Face landmark type.

  Fields:
    position: Face landmark position.
    type: Face landmark type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Face landmark type.

    Values:
      UNKNOWN_LANDMARK: Unknown face landmark detected. Should not be filled.
      LEFT_EYE: Left eye.
      RIGHT_EYE: Right eye.
      LEFT_OF_LEFT_EYEBROW: Left of left eyebrow.
      RIGHT_OF_LEFT_EYEBROW: Right of left eyebrow.
      LEFT_OF_RIGHT_EYEBROW: Left of right eyebrow.
      RIGHT_OF_RIGHT_EYEBROW: Right of right eyebrow.
      MIDPOINT_BETWEEN_EYES: Midpoint between eyes.
      NOSE_TIP: Nose tip.
      UPPER_LIP: Upper lip.
      LOWER_LIP: Lower lip.
      MOUTH_LEFT: Mouth left.
      MOUTH_RIGHT: Mouth right.
      MOUTH_CENTER: Mouth center.
      NOSE_BOTTOM_RIGHT: Nose, bottom right.
      NOSE_BOTTOM_LEFT: Nose, bottom left.
      NOSE_BOTTOM_CENTER: Nose, bottom center.
      LEFT_EYE_TOP_BOUNDARY: Left eye, top boundary.
      LEFT_EYE_RIGHT_CORNER: Left eye, right corner.
      LEFT_EYE_BOTTOM_BOUNDARY: Left eye, bottom boundary.
      LEFT_EYE_LEFT_CORNER: Left eye, left corner.
      RIGHT_EYE_TOP_BOUNDARY: Right eye, top boundary.
      RIGHT_EYE_RIGHT_CORNER: Right eye, right corner.
      RIGHT_EYE_BOTTOM_BOUNDARY: Right eye, bottom boundary.
      RIGHT_EYE_LEFT_CORNER: Right eye, left corner.
      LEFT_EYEBROW_UPPER_MIDPOINT: Left eyebrow, upper midpoint.
      RIGHT_EYEBROW_UPPER_MIDPOINT: Right eyebrow, upper midpoint.
      LEFT_EAR_TRAGION: Left ear tragion.
      RIGHT_EAR_TRAGION: Right ear tragion.
      LEFT_EYE_PUPIL: Left eye pupil.
      RIGHT_EYE_PUPIL: Right eye pupil.
      FOREHEAD_GLABELLA: Forehead glabella.
      CHIN_GNATHION: Chin gnathion.
      CHIN_LEFT_GONION: Chin left gonion.
      CHIN_RIGHT_GONION: Chin right gonion.
      LEFT_CHEEK_CENTER: Left cheek center.
      RIGHT_CHEEK_CENTER: Right cheek center.
    """
        UNKNOWN_LANDMARK = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_OF_LEFT_EYEBROW = 3
        RIGHT_OF_LEFT_EYEBROW = 4
        LEFT_OF_RIGHT_EYEBROW = 5
        RIGHT_OF_RIGHT_EYEBROW = 6
        MIDPOINT_BETWEEN_EYES = 7
        NOSE_TIP = 8
        UPPER_LIP = 9
        LOWER_LIP = 10
        MOUTH_LEFT = 11
        MOUTH_RIGHT = 12
        MOUTH_CENTER = 13
        NOSE_BOTTOM_RIGHT = 14
        NOSE_BOTTOM_LEFT = 15
        NOSE_BOTTOM_CENTER = 16
        LEFT_EYE_TOP_BOUNDARY = 17
        LEFT_EYE_RIGHT_CORNER = 18
        LEFT_EYE_BOTTOM_BOUNDARY = 19
        LEFT_EYE_LEFT_CORNER = 20
        RIGHT_EYE_TOP_BOUNDARY = 21
        RIGHT_EYE_RIGHT_CORNER = 22
        RIGHT_EYE_BOTTOM_BOUNDARY = 23
        RIGHT_EYE_LEFT_CORNER = 24
        LEFT_EYEBROW_UPPER_MIDPOINT = 25
        RIGHT_EYEBROW_UPPER_MIDPOINT = 26
        LEFT_EAR_TRAGION = 27
        RIGHT_EAR_TRAGION = 28
        LEFT_EYE_PUPIL = 29
        RIGHT_EYE_PUPIL = 30
        FOREHEAD_GLABELLA = 31
        CHIN_GNATHION = 32
        CHIN_LEFT_GONION = 33
        CHIN_RIGHT_GONION = 34
        LEFT_CHEEK_CENTER = 35
        RIGHT_CHEEK_CENTER = 36
    position = _messages.MessageField('GoogleCloudVisionV1p2beta1Position', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)