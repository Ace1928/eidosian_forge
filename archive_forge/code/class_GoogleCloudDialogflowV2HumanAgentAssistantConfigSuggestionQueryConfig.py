from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfig(_messages.Message):
    """Config for suggestion query.

  Fields:
    confidenceThreshold: Confidence threshold of query result. Agent Assist
      gives each suggestion a score in the range [0.0, 1.0], based on the
      relevance between the suggestion and the current conversation context. A
      score of 0.0 has no relevance, while a score of 1.0 has high relevance.
      Only suggestions with a score greater than or equal to the value of this
      field are included in the results. For a baseline model (the default),
      the recommended value is in the range [0.05, 0.1]. For a custom model,
      there is no recommended value. Tune this value by starting from a very
      low value and slowly increasing until you have desired results. If this
      field is not set, it defaults to 0.0, which means that all suggestions
      are returned. Supported features: ARTICLE_SUGGESTION, FAQ, SMART_REPLY,
      SMART_COMPOSE, KNOWLEDGE_SEARCH, KNOWLEDGE_ASSIST, ENTITY_EXTRACTION.
    contextFilterSettings: Determines how recent conversation context is
      filtered when generating suggestions. If unspecified, no messages will
      be dropped.
    dialogflowQuerySource: Query from Dialogflow agent. It is used by
      DIALOGFLOW_ASSIST.
    documentQuerySource: Query from knowledge base document. It is used by:
      SMART_REPLY, SMART_COMPOSE.
    knowledgeBaseQuerySource: Query from knowledgebase. It is used by:
      ARTICLE_SUGGESTION, FAQ.
    maxResults: Maximum number of results to return. Currently, if unset,
      defaults to 10. And the max number is 20.
    sections: Optional. The customized sections chosen to return when
      requesting a summary of a conversation.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    contextFilterSettings = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigContextFilterSettings', 2)
    dialogflowQuerySource = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigDialogflowQuerySource', 3)
    documentQuerySource = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigDocumentQuerySource', 4)
    knowledgeBaseQuerySource = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigKnowledgeBaseQuerySource', 5)
    maxResults = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    sections = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigSections', 7)