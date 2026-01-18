from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsCreateRequest object.

  Fields:
    googleCloudAiplatformV1IndexEndpoint: A
      GoogleCloudAiplatformV1IndexEndpoint resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to create the
      IndexEndpoint in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1IndexEndpoint = _messages.MessageField('GoogleCloudAiplatformV1IndexEndpoint', 1)
    parent = _messages.StringField(2, required=True)