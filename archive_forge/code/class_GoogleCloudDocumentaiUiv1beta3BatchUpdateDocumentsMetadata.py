from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3BatchUpdateDocumentsMetadata(_messages.Message):
    """A GoogleCloudDocumentaiUiv1beta3BatchUpdateDocumentsMetadata object.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    individualBatchUpdateStatuses: The list of response details of each
      document.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    individualBatchUpdateStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3BatchUpdateDocumentsMetadataIndividualBatchUpdateStatus', 2, repeated=True)