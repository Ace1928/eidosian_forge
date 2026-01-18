from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DeployConversationModelOperationMetadata(_messages.Message):
    """Metadata for a ConversationModels.DeployConversationModel operation.

  Fields:
    conversationModel: The resource name of the conversation model. Format:
      `projects//conversationModels/`
    createTime: Timestamp when request to deploy conversation model was
      submitted. The time is measured on server side.
  """
    conversationModel = _messages.StringField(1)
    createTime = _messages.StringField(2)