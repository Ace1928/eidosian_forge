from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1SuggestionResult(_messages.Message):
    """One response of different type of suggestion response which is used in
  the response of Participants.AnalyzeContent and Participants.AnalyzeContent,
  as well as HumanAgentAssistantEvent.

  Fields:
    error: Error status if the request failed.
    suggestArticlesResponse: SuggestArticlesResponse if request is for
      ARTICLE_SUGGESTION.
    suggestDialogflowAssistsResponse: SuggestDialogflowAssistsResponse if
      request is for DIALOGFLOW_ASSIST.
    suggestEntityExtractionResponse: SuggestDialogflowAssistsResponse if
      request is for ENTITY_EXTRACTION.
    suggestFaqAnswersResponse: SuggestFaqAnswersResponse if request is for
      FAQ_ANSWER.
    suggestSmartRepliesResponse: SuggestSmartRepliesResponse if request is for
      SMART_REPLY.
  """
    error = _messages.MessageField('GoogleRpcStatus', 1)
    suggestArticlesResponse = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestArticlesResponse', 2)
    suggestDialogflowAssistsResponse = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestDialogflowAssistsResponse', 3)
    suggestEntityExtractionResponse = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestDialogflowAssistsResponse', 4)
    suggestFaqAnswersResponse = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestFaqAnswersResponse', 5)
    suggestSmartRepliesResponse = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestSmartRepliesResponse', 6)