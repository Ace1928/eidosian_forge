from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ResyncDatasetMetadata(_messages.Message):
    """The metadata proto of `ResyncDataset` method.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    datasetResyncStatuses: The list of dataset resync statuses. Not checked
      when ResyncDatasetRequest.dataset_documents is specified.
    individualDocumentResyncStatuses: The list of document resync statuses.
      The same document could have multiple
      `individual_document_resync_statuses` if it has multiple
      inconsistencies.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    datasetResyncStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ResyncDatasetMetadataDatasetResyncStatus', 2, repeated=True)
    individualDocumentResyncStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ResyncDatasetMetadataIndividualDocumentResyncStatus', 3, repeated=True)