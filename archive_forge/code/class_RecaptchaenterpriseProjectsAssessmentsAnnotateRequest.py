from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsAssessmentsAnnotateRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsAssessmentsAnnotateRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest: A
      GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest resource to be
      passed as the request body.
    name: Required. The resource name of the Assessment, in the format
      `projects/{project}/assessments/{assessment}`.
  """
    googleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest', 1)
    name = _messages.StringField(2, required=True)