from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsSuggestionsSuggestConversationSummaryRequest(_messages.Message):
    """A
  DialogflowProjectsConversationsSuggestionsSuggestConversationSummaryRequest
  object.

  Fields:
    conversation: Required. The conversation to fetch suggestion for. Format:
      `projects//locations//conversations/`.
    googleCloudDialogflowV2SuggestConversationSummaryRequest: A
      GoogleCloudDialogflowV2SuggestConversationSummaryRequest resource to be
      passed as the request body.
  """
    conversation = _messages.StringField(1, required=True)
    googleCloudDialogflowV2SuggestConversationSummaryRequest = _messages.MessageField('GoogleCloudDialogflowV2SuggestConversationSummaryRequest', 2)