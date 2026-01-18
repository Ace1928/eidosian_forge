from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ExportDocumentsMetadataIndividualExportStatus(_messages.Message):
    """The status of each individual document in the export process.

  Fields:
    documentId: The path to source docproto of the document.
    outputGcsDestination: The output_gcs_destination of the exported document
      if it was successful, otherwise empty.
    status: The status of the exporting of the document.
  """
    documentId = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3DocumentId', 1)
    outputGcsDestination = _messages.StringField(2)
    status = _messages.MessageField('GoogleRpcStatus', 3)