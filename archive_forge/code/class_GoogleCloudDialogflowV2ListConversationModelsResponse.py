from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListConversationModelsResponse(_messages.Message):
    """The response message for ConversationModels.ListConversationModels

  Fields:
    conversationModels: The list of models to return.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    conversationModels = _messages.MessageField('GoogleCloudDialogflowV2ConversationModel', 1, repeated=True)
    nextPageToken = _messages.StringField(2)