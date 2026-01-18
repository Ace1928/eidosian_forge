from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendation(_messages.Message):
    """The message format of a recommendation from the assessment.

  Enums:
    VerdictValueValuesEnum: Verdict indicates the assessment result.
    WeightValueValuesEnum: The weight of the assessment which was set in the
      profile.

  Fields:
    displayName: The display name of the assessment.
    recommendations: The recommended steps of the assessment.
    scoreImpact: Score impact indicates the impact on the overall score if the
      assessment were to pass.
    verdict: Verdict indicates the assessment result.
    weight: The weight of the assessment which was set in the profile.
  """

    class VerdictValueValuesEnum(_messages.Enum):
        """Verdict indicates the assessment result.

    Values:
      VERDICT_UNSPECIFIED: The verdict is unspecified.
      PASS: The assessment has passed.
      FAIL: The assessment has failed.
    """
        VERDICT_UNSPECIFIED = 0
        PASS = 1
        FAIL = 2

    class WeightValueValuesEnum(_messages.Enum):
        """The weight of the assessment which was set in the profile.

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
    displayName = _messages.StringField(1)
    recommendations = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendationRecommendation', 2, repeated=True)
    scoreImpact = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    verdict = _messages.EnumField('VerdictValueValuesEnum', 4)
    weight = _messages.EnumField('WeightValueValuesEnum', 5)