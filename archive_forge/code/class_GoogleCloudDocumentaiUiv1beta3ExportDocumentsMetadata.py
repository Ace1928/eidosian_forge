from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ExportDocumentsMetadata(_messages.Message):
    """Metadata of the batch export documents operation.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    individualExportStatuses: The list of response details of each document.
    splitExportStats: The list of statistics for each dataset split type.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    individualExportStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ExportDocumentsMetadataIndividualExportStatus', 2, repeated=True)
    splitExportStats = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ExportDocumentsMetadataSplitExportStat', 3, repeated=True)