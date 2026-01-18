from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsChatCompletionsRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsChatCompletionsRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/endpoints/openapi`
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 2)