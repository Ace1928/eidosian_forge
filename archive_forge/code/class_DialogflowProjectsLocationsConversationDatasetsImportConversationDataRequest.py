from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationDatasetsImportConversationDataRequest(_messages.Message):
    """A
  DialogflowProjectsLocationsConversationDatasetsImportConversationDataRequest
  object.

  Fields:
    googleCloudDialogflowV2ImportConversationDataRequest: A
      GoogleCloudDialogflowV2ImportConversationDataRequest resource to be
      passed as the request body.
    name: Required. Dataset resource name. Format:
      `projects//locations//conversationDatasets/`
  """
    googleCloudDialogflowV2ImportConversationDataRequest = _messages.MessageField('GoogleCloudDialogflowV2ImportConversationDataRequest', 1)
    name = _messages.StringField(2, required=True)