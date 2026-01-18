from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPublishersModelsRawPredictRequest(_messages.Message):
    """A AiplatformProjectsLocationsPublishersModelsRawPredictRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1RawPredictRequest: A
      GoogleCloudAiplatformV1RawPredictRequest resource to be passed as the
      request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1RawPredictRequest = _messages.MessageField('GoogleCloudAiplatformV1RawPredictRequest', 2)