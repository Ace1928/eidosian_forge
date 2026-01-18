from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SearchKnowledgeAnswerAnswerSource(_messages.Message):
    """The sources of the answers.

  Fields:
    snippet: The relevant snippet of the article.
    title: The title of the article.
    uri: The URI of the article.
  """
    snippet = _messages.StringField(1)
    title = _messages.StringField(2)
    uri = _messages.StringField(3)