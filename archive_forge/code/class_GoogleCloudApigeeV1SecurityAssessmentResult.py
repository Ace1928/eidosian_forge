from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAssessmentResult(_messages.Message):
    """The security assessment result for one resource.

  Fields:
    createTime: The time of the assessment of this resource. This could lag
      behind `assessment_time` due to caching within the backend.
    error: The error status if scoring fails.
    resource: The assessed resource.
    scoringResult: The result of the assessment.
  """
    createTime = _messages.StringField(1)
    error = _messages.MessageField('GoogleRpcStatus', 2)
    resource = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResultResource', 3)
    scoringResult = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResultScoringResult', 4)