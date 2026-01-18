from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentKnowledgeBasesDocumentsDeleteRequest(_messages.Message):
    """A DialogflowProjectsAgentKnowledgeBasesDocumentsDeleteRequest object.

  Fields:
    name: Required. The name of the document to delete. Format:
      `projects//locations//knowledgeBases//documents/`.
  """
    name = _messages.StringField(1, required=True)