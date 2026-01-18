from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListKnowledgeBasesResponse(_messages.Message):
    """Response message for KnowledgeBases.ListKnowledgeBases.

  Fields:
    knowledgeBases: The list of knowledge bases.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    knowledgeBases = _messages.MessageField('GoogleCloudDialogflowV2KnowledgeBase', 1, repeated=True)
    nextPageToken = _messages.StringField(2)