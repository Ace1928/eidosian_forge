from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsCheckTrialEarlyStoppingStateRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsStudiesTrialsCheckTrialEarlyStoppingStateRequest
  object.

  Fields:
    googleCloudAiplatformV1CheckTrialEarlyStoppingStateRequest: A
      GoogleCloudAiplatformV1CheckTrialEarlyStoppingStateRequest resource to
      be passed as the request body.
    trialName: Required. The Trial's name. Format:
      `projects/{project}/locations/{location}/studies/{study}/trials/{trial}`
  """
    googleCloudAiplatformV1CheckTrialEarlyStoppingStateRequest = _messages.MessageField('GoogleCloudAiplatformV1CheckTrialEarlyStoppingStateRequest', 1)
    trialName = _messages.StringField(2, required=True)