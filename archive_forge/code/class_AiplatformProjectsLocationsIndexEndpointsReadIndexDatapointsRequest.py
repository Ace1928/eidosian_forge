from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsReadIndexDatapointsRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsReadIndexDatapointsRequest
  object.

  Fields:
    googleCloudAiplatformV1beta1ReadIndexDatapointsRequest: A
      GoogleCloudAiplatformV1beta1ReadIndexDatapointsRequest resource to be
      passed as the request body.
    indexEndpoint: Required. The name of the index endpoint. Format:
      `projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}
      `
  """
    googleCloudAiplatformV1beta1ReadIndexDatapointsRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ReadIndexDatapointsRequest', 1)
    indexEndpoint = _messages.StringField(2, required=True)