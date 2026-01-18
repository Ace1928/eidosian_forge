from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QueryExtensionResponse(_messages.Message):
    """Response message for ExtensionExecutionService.QueryExtension.

  Fields:
    failureMessage: Failure message if any.
    steps: Steps of extension or LLM interaction, can contain function call,
      function response, or text response. The last step contains the final
      response to the query.
  """
    failureMessage = _messages.StringField(1)
    steps = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 2, repeated=True)