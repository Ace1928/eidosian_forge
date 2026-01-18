from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentKnowledgeBasesDocumentsCreateRequest(_messages.Message):
    """A DialogflowProjectsAgentKnowledgeBasesDocumentsCreateRequest object.

  Fields:
    googleCloudDialogflowV2Document: A GoogleCloudDialogflowV2Document
      resource to be passed as the request body.
    parent: Required. The knowledge base to create a document for. Format:
      `projects//locations//knowledgeBases/`.
  """
    googleCloudDialogflowV2Document = _messages.MessageField('GoogleCloudDialogflowV2Document', 1)
    parent = _messages.StringField(2, required=True)