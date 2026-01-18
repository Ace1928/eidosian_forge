from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsReasoningEnginesQueryRequest(_messages.Message):
    """A AiplatformProjectsLocationsReasoningEnginesQueryRequest object.

  Fields:
    googleCloudAiplatformV1beta1QueryReasoningEngineRequest: A
      GoogleCloudAiplatformV1beta1QueryReasoningEngineRequest resource to be
      passed as the request body.
    name: Required. The name of the ReasoningEngine resource to use. Format: `
      projects/{project}/locations/{location}/reasoningEngines/{reasoning_engi
      ne}`
  """
    googleCloudAiplatformV1beta1QueryReasoningEngineRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1QueryReasoningEngineRequest', 1)
    name = _messages.StringField(2, required=True)