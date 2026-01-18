from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationDataset(_messages.Message):
    """Represents a conversation dataset that a user imports raw data into. The
  data inside ConversationDataset can not be changed after
  ImportConversationData finishes (and calling ImportConversationData on a
  dataset that already has data is not allowed).

  Fields:
    conversationCount: Output only. The number of conversations this
      conversation dataset contains.
    conversationInfo: Output only. Metadata set during conversation data
      import.
    createTime: Output only. Creation time of this dataset.
    description: Optional. The description of the dataset. Maximum of 10000
      bytes.
    displayName: Required. The display name of the dataset. Maximum of 64
      bytes.
    inputConfig: Output only. Input configurations set during conversation
      data import.
    name: Output only. ConversationDataset resource name. Format:
      `projects//locations//conversationDatasets/`
  """
    conversationCount = _messages.IntegerField(1)
    conversationInfo = _messages.MessageField('GoogleCloudDialogflowV2ConversationInfo', 2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    inputConfig = _messages.MessageField('GoogleCloudDialogflowV2InputConfig', 6)
    name = _messages.StringField(7)