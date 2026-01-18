from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentKnowledgeBasesDocumentsPatchRequest(_messages.Message):
    """A DialogflowProjectsAgentKnowledgeBasesDocumentsPatchRequest object.

  Fields:
    googleCloudDialogflowV2Document: A GoogleCloudDialogflowV2Document
      resource to be passed as the request body.
    name: Optional. The document resource name. The name must be empty when
      creating a document. Format:
      `projects//locations//knowledgeBases//documents/`.
    updateMask: Optional. Not specified means `update all`. Currently, only
      `display_name` can be updated, an InvalidArgument will be returned for
      attempting to update other fields.
  """
    googleCloudDialogflowV2Document = _messages.MessageField('GoogleCloudDialogflowV2Document', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)