from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsPatchRequest object.

  Fields:
    googleCloudAiplatformV1IndexEndpoint: A
      GoogleCloudAiplatformV1IndexEndpoint resource to be passed as the
      request body.
    name: Output only. The resource name of the IndexEndpoint.
    updateMask: Required. The update mask applies to the resource. See
      google.protobuf.FieldMask.
  """
    googleCloudAiplatformV1IndexEndpoint = _messages.MessageField('GoogleCloudAiplatformV1IndexEndpoint', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)