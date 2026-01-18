from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsMutateDeployedModelRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsMutateDeployedModelRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint resource into which to mutate
      a DeployedModel. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1MutateDeployedModelRequest: A
      GoogleCloudAiplatformV1MutateDeployedModelRequest resource to be passed
      as the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1MutateDeployedModelRequest = _messages.MessageField('GoogleCloudAiplatformV1MutateDeployedModelRequest', 2)