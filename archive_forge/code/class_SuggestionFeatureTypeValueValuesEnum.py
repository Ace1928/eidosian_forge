from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuggestionFeatureTypeValueValuesEnum(_messages.Enum):
    """Required. The type of the suggestion feature to add or update.

    Values:
      TYPE_UNSPECIFIED: Unspecified feature type.
      ARTICLE_SUGGESTION: Run article suggestion model for chat.
      FAQ: Run FAQ model.
      SMART_REPLY: Run smart reply model for chat.
      DIALOGFLOW_ASSIST: Run Dialogflow assist model for chat, which will
        return automated agent response as suggestion.
      CONVERSATION_SUMMARIZATION: Run conversation summarization model for
        chat.
      KNOWLEDGE_SEARCH: Run knowledge search with text input from agent or
        text generated query.
    """
    TYPE_UNSPECIFIED = 0
    ARTICLE_SUGGESTION = 1
    FAQ = 2
    SMART_REPLY = 3
    DIALOGFLOW_ASSIST = 4
    CONVERSATION_SUMMARIZATION = 5
    KNOWLEDGE_SEARCH = 6