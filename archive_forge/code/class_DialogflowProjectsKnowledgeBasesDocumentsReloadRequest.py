from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsKnowledgeBasesDocumentsReloadRequest(_messages.Message):
    """A DialogflowProjectsKnowledgeBasesDocumentsReloadRequest object.

  Fields:
    googleCloudDialogflowV2ReloadDocumentRequest: A
      GoogleCloudDialogflowV2ReloadDocumentRequest resource to be passed as
      the request body.
    name: Required. The name of the document to reload. Format:
      `projects//locations//knowledgeBases//documents/`
  """
    googleCloudDialogflowV2ReloadDocumentRequest = _messages.MessageField('GoogleCloudDialogflowV2ReloadDocumentRequest', 1)
    name = _messages.StringField(2, required=True)