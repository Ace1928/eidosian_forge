from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionFeatureConfig(_messages.Message):
    """Config for suggestion features.

  Fields:
    conversationModelConfig: Configs of custom conversation model.
    conversationProcessConfig: Configs for processing conversation.
    disableAgentQueryLogging: Optional. Disable the logging of search queries
      sent by human agents. It can prevent those queries from being stored at
      answer records. Supported features: KNOWLEDGE_SEARCH.
    enableConversationAugmentedQuery: Optional. Enable including conversation
      context during query answer generation. Supported features:
      KNOWLEDGE_SEARCH.
    enableEventBasedSuggestion: Automatically iterates all participants and
      tries to compile suggestions. Supported features: ARTICLE_SUGGESTION,
      FAQ, DIALOGFLOW_ASSIST, KNOWLEDGE_ASSIST.
    queryConfig: Configs of query.
    suggestionFeature: The suggestion feature.
    suggestionTriggerSettings: Settings of suggestion trigger. Currently, only
      ARTICLE_SUGGESTION and FAQ will use this field.
  """
    conversationModelConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigConversationModelConfig', 1)
    conversationProcessConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigConversationProcessConfig', 2)
    disableAgentQueryLogging = _messages.BooleanField(3)
    enableConversationAugmentedQuery = _messages.BooleanField(4)
    enableEventBasedSuggestion = _messages.BooleanField(5)
    queryConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfig', 6)
    suggestionFeature = _messages.MessageField('GoogleCloudDialogflowV2SuggestionFeature', 7)
    suggestionTriggerSettings = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionTriggerSettings', 8)