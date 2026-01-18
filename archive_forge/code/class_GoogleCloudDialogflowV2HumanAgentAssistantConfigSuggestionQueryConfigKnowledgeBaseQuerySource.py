from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigKnowledgeBaseQuerySource(_messages.Message):
    """Knowledge base source settings. Supported features: ARTICLE_SUGGESTION,
  FAQ.

  Fields:
    knowledgeBases: Required. Knowledge bases to query. Format:
      `projects//locations//knowledgeBases/`. Currently, at most 5 knowledge
      bases are supported.
  """
    knowledgeBases = _messages.StringField(1, repeated=True)