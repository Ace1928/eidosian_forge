from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3BatchUpdateDocumentsMetadataIndividualBatchUpdateStatus(_messages.Message):
    """The status of each individual document in the batch update process.

  Fields:
    documentId: The document id of the document.
    status: The status of updating the document in storage.
  """
    documentId = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3DocumentId', 1)
    status = _messages.MessageField('GoogleRpcStatus', 2)