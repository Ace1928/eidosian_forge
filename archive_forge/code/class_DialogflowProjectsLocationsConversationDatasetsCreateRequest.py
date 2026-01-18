from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationDatasetsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationDatasetsCreateRequest object.

  Fields:
    googleCloudDialogflowV2ConversationDataset: A
      GoogleCloudDialogflowV2ConversationDataset resource to be passed as the
      request body.
    parent: Required. The project to create conversation dataset for. Format:
      `projects//locations/`
  """
    googleCloudDialogflowV2ConversationDataset = _messages.MessageField('GoogleCloudDialogflowV2ConversationDataset', 1)
    parent = _messages.StringField(2, required=True)