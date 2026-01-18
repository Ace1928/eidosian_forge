from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAssessmentResultScoringResult(_messages.Message):
    """The result of the assessment.

  Enums:
    SeverityValueValuesEnum: The severity of the assessment.

  Messages:
    AssessmentRecommendationsValue: The recommendations of the assessment. The
      key is the "name" of the assessment (not display_name), and the value
      are the recommendations.
    FailedAssessmentPerWeightValue: The number of failed assessments grouped
      by its weight. Keys are one of the following: "MAJOR", "MODERATE",
      "MINOR".

  Fields:
    assessmentRecommendations: The recommendations of the assessment. The key
      is the "name" of the assessment (not display_name), and the value are
      the recommendations.
    dataUpdateTime: The time when resource data was last fetched for this
      resource. This time may be different than when the resource was actually
      updated due to lag in data collection.
    failedAssessmentPerWeight: The number of failed assessments grouped by its
      weight. Keys are one of the following: "MAJOR", "MODERATE", "MINOR".
    score: The security score of the assessment.
    severity: The severity of the assessment.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the assessment.

    Values:
      SEVERITY_UNSPECIFIED: Severity is not defined.
      LOW: Severity is low.
      MEDIUM: Severity is medium.
      HIGH: Severity is high.
      NONE: Severity is none.
      NO_RISK: Severity represents no risk
      MINIMAL: Severity is minimal
    """
        SEVERITY_UNSPECIFIED = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        NONE = 4
        NO_RISK = 5
        MINIMAL = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AssessmentRecommendationsValue(_messages.Message):
        """The recommendations of the assessment. The key is the "name" of the
    assessment (not display_name), and the value are the recommendations.

    Messages:
      AdditionalProperty: An additional property for a
        AssessmentRecommendationsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AssessmentRecommendationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AssessmentRecommendationsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAsses
          smentRecommendation attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendation', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FailedAssessmentPerWeightValue(_messages.Message):
        """The number of failed assessments grouped by its weight. Keys are one
    of the following: "MAJOR", "MODERATE", "MINOR".

    Messages:
      AdditionalProperty: An additional property for a
        FailedAssessmentPerWeightValue object.

    Fields:
      additionalProperties: Additional properties of type
        FailedAssessmentPerWeightValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FailedAssessmentPerWeightValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    assessmentRecommendations = _messages.MessageField('AssessmentRecommendationsValue', 1)
    dataUpdateTime = _messages.StringField(2)
    failedAssessmentPerWeight = _messages.MessageField('FailedAssessmentPerWeightValue', 3)
    score = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    severity = _messages.EnumField('SeverityValueValuesEnum', 5)