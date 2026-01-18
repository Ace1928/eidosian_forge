from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigMessageAnalysisConfig(_messages.Message):
    """Configuration for analyses to run on each conversation message.

  Fields:
    enableEntityExtraction: Enable entity extraction in conversation messages
      on [agent assist
      stage](https://cloud.google.com/dialogflow/priv/docs/contact-
      center/basics#stages). If unspecified, defaults to false. Currently,
      this feature is not general available, please contact Google to get
      access.
    enableSentimentAnalysis: Enable sentiment analysis in conversation
      messages on [agent assist
      stage](https://cloud.google.com/dialogflow/priv/docs/contact-
      center/basics#stages). If unspecified, defaults to false. Sentiment
      analysis inspects user input and identifies the prevailing subjective
      opinion, especially to determine a user's attitude as positive,
      negative, or neutral: https://cloud.google.com/natural-
      language/docs/basics#sentiment_analysis For
      Participants.StreamingAnalyzeContent method, result will be in
      StreamingAnalyzeContentResponse.message.SentimentAnalysisResult. For
      Participants.AnalyzeContent method, result will be in
      AnalyzeContentResponse.message.SentimentAnalysisResult For
      Conversations.ListMessages method, result will be in
      ListMessagesResponse.messages.SentimentAnalysisResult If Pub/Sub
      notification is configured, result will be in
      ConversationEvent.new_message_payload.SentimentAnalysisResult.
  """
    enableEntityExtraction = _messages.BooleanField(1)
    enableSentimentAnalysis = _messages.BooleanField(2)