from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEntityTypesListRequest(_messages.Message):
    """A DialogflowProjectsAgentEntityTypesListRequest object.

  Fields:
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The agent to list all entity types from. Format:
      `projects//agent`.
  """
    languageCode = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)