from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SuggestionResult(_messages.Message):
    """One response of different type of suggestion response which is used in
  the response of Participants.AnalyzeContent and Participants.AnalyzeContent,
  as well as HumanAgentAssistantEvent.

  Fields:
    error: Error status if the request failed.
    suggestArticlesResponse: SuggestArticlesResponse if request is for
      ARTICLE_SUGGESTION.
    suggestFaqAnswersResponse: SuggestFaqAnswersResponse if request is for
      FAQ_ANSWER.
    suggestSmartRepliesResponse: SuggestSmartRepliesResponse if request is for
      SMART_REPLY.
  """
    error = _messages.MessageField('GoogleRpcStatus', 1)
    suggestArticlesResponse = _messages.MessageField('GoogleCloudDialogflowV2SuggestArticlesResponse', 2)
    suggestFaqAnswersResponse = _messages.MessageField('GoogleCloudDialogflowV2SuggestFaqAnswersResponse', 3)
    suggestSmartRepliesResponse = _messages.MessageField('GoogleCloudDialogflowV2SuggestSmartRepliesResponse', 4)