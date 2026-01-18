from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SentimentAnalysisResult(_messages.Message):
    """The result of sentiment analysis. Sentiment analysis inspects user input
  and identifies the prevailing subjective opinion, especially to determine a
  user's attitude as positive, negative, or neutral. For
  Participants.DetectIntent, it needs to be configured in
  DetectIntentRequest.query_params. For Participants.StreamingDetectIntent, it
  needs to be configured in StreamingDetectIntentRequest.query_params. And for
  Participants.AnalyzeContent and Participants.StreamingAnalyzeContent, it
  needs to be configured in ConversationProfile.human_agent_assistant_config

  Fields:
    queryTextSentiment: The sentiment analysis result for `query_text`.
  """
    queryTextSentiment = _messages.MessageField('GoogleCloudDialogflowV2Sentiment', 1)