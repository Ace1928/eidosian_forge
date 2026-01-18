from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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