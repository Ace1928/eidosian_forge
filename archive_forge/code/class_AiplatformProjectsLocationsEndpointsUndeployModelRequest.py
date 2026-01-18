from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsUndeployModelRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsUndeployModelRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint resource from which to
      undeploy a Model. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1UndeployModelRequest: A
      GoogleCloudAiplatformV1UndeployModelRequest resource to be passed as the
      request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1UndeployModelRequest = _messages.MessageField('GoogleCloudAiplatformV1UndeployModelRequest', 2)