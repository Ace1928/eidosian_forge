from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3BatchDeleteDocumentsMetadata(_messages.Message):
    """A GoogleCloudDocumentaiUiv1beta3BatchDeleteDocumentsMetadata object.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    errorDocumentCount: Total number of documents that failed to be deleted in
      storage.
    individualBatchDeleteStatuses: The list of response details of each
      document.
    totalDocumentCount: Total number of documents deleting from dataset.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    errorDocumentCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    individualBatchDeleteStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3BatchDeleteDocumentsMetadataIndividualBatchDeleteStatus', 3, repeated=True)
    totalDocumentCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)