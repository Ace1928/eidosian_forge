from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsDirectRawPredictRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsDirectRawPredictRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1beta1DirectRawPredictRequest: A
      GoogleCloudAiplatformV1beta1DirectRawPredictRequest resource to be
      passed as the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1DirectRawPredictRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1DirectRawPredictRequest', 2)