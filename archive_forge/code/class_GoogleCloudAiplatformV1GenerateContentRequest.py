from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GenerateContentRequest(_messages.Message):
    """Request message for [PredictionService.GenerateContent].

  Fields:
    contents: Required. The content of the current conversation with the
      model. For single-turn queries, this is a single instance. For multi-
      turn queries, this is a repeated field that contains conversation
      history + latest request.
    generationConfig: Optional. Generation config.
    safetySettings: Optional. Per request settings for blocking unsafe
      content. Enforced on GenerateContentResponse.candidates.
    systemInstructions: Optional. The user provided system instructions for
      the model. Note: only text should be used in parts.
    tools: Optional. A list of `Tools` the model may use to generate the next
      response. A `Tool` is a piece of code that enables the system to
      interact with external systems to perform an action, or set of actions,
      outside of knowledge and scope of the model.
  """
    contents = _messages.MessageField('GoogleCloudAiplatformV1Content', 1, repeated=True)
    generationConfig = _messages.MessageField('GoogleCloudAiplatformV1GenerationConfig', 2)
    safetySettings = _messages.MessageField('GoogleCloudAiplatformV1SafetySetting', 3, repeated=True)
    systemInstructions = _messages.MessageField('GoogleCloudAiplatformV1Content', 4, repeated=True)
    tools = _messages.MessageField('GoogleCloudAiplatformV1Tool', 5, repeated=True)