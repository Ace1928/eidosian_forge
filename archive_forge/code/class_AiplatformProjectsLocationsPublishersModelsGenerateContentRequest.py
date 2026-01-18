from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPublishersModelsGenerateContentRequest(_messages.Message):
    """A AiplatformProjectsLocationsPublishersModelsGenerateContentRequest
  object.

  Fields:
    googleCloudAiplatformV1GenerateContentRequest: A
      GoogleCloudAiplatformV1GenerateContentRequest resource to be passed as
      the request body.
    model: Required. The name of the publisher model requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/publishers/*/models/*`
  """
    googleCloudAiplatformV1GenerateContentRequest = _messages.MessageField('GoogleCloudAiplatformV1GenerateContentRequest', 1)
    model = _messages.StringField(2, required=True)