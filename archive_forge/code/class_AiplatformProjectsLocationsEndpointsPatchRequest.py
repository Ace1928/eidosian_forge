from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsPatchRequest object.

  Fields:
    googleCloudAiplatformV1Endpoint: A GoogleCloudAiplatformV1Endpoint
      resource to be passed as the request body.
    name: Output only. The resource name of the Endpoint.
    updateMask: Required. The update mask applies to the resource. See
      google.protobuf.FieldMask.
  """
    googleCloudAiplatformV1Endpoint = _messages.MessageField('GoogleCloudAiplatformV1Endpoint', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)