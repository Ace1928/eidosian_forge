from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsStreamRawPredictRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsStreamRawPredictRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1StreamRawPredictRequest: A
      GoogleCloudAiplatformV1StreamRawPredictRequest resource to be passed as
      the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1StreamRawPredictRequest = _messages.MessageField('GoogleCloudAiplatformV1StreamRawPredictRequest', 2)