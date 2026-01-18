from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListConversationModelEvaluationsResponse(_messages.Message):
    """The response message for
  ConversationModels.ListConversationModelEvaluations

  Fields:
    conversationModelEvaluations: The list of evaluations to return.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    conversationModelEvaluations = _messages.MessageField('GoogleCloudDialogflowV2ConversationModelEvaluation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)