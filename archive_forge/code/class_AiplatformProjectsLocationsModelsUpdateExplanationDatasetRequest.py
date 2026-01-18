from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsUpdateExplanationDatasetRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsUpdateExplanationDatasetRequest
  object.

  Fields:
    googleCloudAiplatformV1UpdateExplanationDatasetRequest: A
      GoogleCloudAiplatformV1UpdateExplanationDatasetRequest resource to be
      passed as the request body.
    model: Required. The resource name of the Model to update. Format:
      `projects/{project}/locations/{location}/models/{model}`
  """
    googleCloudAiplatformV1UpdateExplanationDatasetRequest = _messages.MessageField('GoogleCloudAiplatformV1UpdateExplanationDatasetRequest', 1)
    model = _messages.StringField(2, required=True)