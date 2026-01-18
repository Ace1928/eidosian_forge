from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsExplainRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsExplainRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      explanation. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1ExplainRequest: A
      GoogleCloudAiplatformV1ExplainRequest resource to be passed as the
      request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1ExplainRequest = _messages.MessageField('GoogleCloudAiplatformV1ExplainRequest', 2)