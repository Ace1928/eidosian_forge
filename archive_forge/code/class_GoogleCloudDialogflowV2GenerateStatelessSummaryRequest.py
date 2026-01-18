from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GenerateStatelessSummaryRequest(_messages.Message):
    """The request message for Conversations.GenerateStatelessSummary.

  Fields:
    conversationProfile: Required. A ConversationProfile containing
      information required for Summary generation. Required fields:
      {language_code, security_settings} Optional fields:
      {agent_assistant_config}
    latestMessage: The name of the latest conversation message used as context
      for generating a Summary. If empty, the latest message of the
      conversation will be used. The format is specific to the user and the
      names of the messages provided.
    maxContextSize: Max number of messages prior to and including
      [latest_message] to use as context when compiling the suggestion. By
      default 500 and at most 1000.
    statelessConversation: Required. The conversation to suggest a summary
      for.
  """
    conversationProfile = _messages.MessageField('GoogleCloudDialogflowV2ConversationProfile', 1)
    latestMessage = _messages.StringField(2)
    maxContextSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    statelessConversation = _messages.MessageField('GoogleCloudDialogflowV2GenerateStatelessSummaryRequestMinimalConversation', 4)