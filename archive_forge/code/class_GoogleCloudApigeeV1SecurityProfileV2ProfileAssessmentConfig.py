from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityProfileV2ProfileAssessmentConfig(_messages.Message):
    """The configuration definition for a specific assessment.

  Enums:
    WeightValueValuesEnum: The weight of the assessment.

  Fields:
    weight: The weight of the assessment.
  """

    class WeightValueValuesEnum(_messages.Enum):
        """The weight of the assessment.

    Values:
      WEIGHT_UNSPECIFIED: The weight is unspecified.
      MINOR: The weight is minor.
      MODERATE: The weight is moderate.
      MAJOR: The weight is major.
    """
        WEIGHT_UNSPECIFIED = 0
        MINOR = 1
        MODERATE = 2
        MAJOR = 3
    weight = _messages.EnumField('WeightValueValuesEnum', 1)