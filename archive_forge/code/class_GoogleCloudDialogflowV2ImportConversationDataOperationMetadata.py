from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ImportConversationDataOperationMetadata(_messages.Message):
    """Metadata for a ConversationDatasets.ImportConversationData operation.

  Fields:
    conversationDataset: The resource name of the imported conversation
      dataset. Format: `projects//locations//conversationDatasets/`
    createTime: Timestamp when import conversation data request was created.
      The time is measured on server side.
    partialFailures: Partial failures are failures that don't fail the whole
      long running operation, e.g. single files that couldn't be read.
  """
    conversationDataset = _messages.StringField(1)
    createTime = _messages.StringField(2)
    partialFailures = _messages.MessageField('GoogleRpcStatus', 3, repeated=True)