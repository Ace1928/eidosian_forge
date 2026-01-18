from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2CreateConversationModelEvaluationRequest(_messages.Message):
    """The request message for
  ConversationModels.CreateConversationModelEvaluation

  Fields:
    conversationModelEvaluation: Required. The conversation model evaluation
      to be created.
  """
    conversationModelEvaluation = _messages.MessageField('GoogleCloudDialogflowV2ConversationModelEvaluation', 1)