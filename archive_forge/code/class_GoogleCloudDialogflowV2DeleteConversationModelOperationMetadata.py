from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DeleteConversationModelOperationMetadata(_messages.Message):
    """Metadata for a ConversationModels.DeleteConversationModel operation.

  Fields:
    conversationModel: The resource name of the conversation model. Format:
      `projects//conversationModels/`
    createTime: Timestamp when delete conversation model request was created.
      The time is measured on server side.
  """
    conversationModel = _messages.StringField(1)
    createTime = _messages.StringField(2)