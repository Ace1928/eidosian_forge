from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ImportDocumentsRequest(_messages.Message):
    """Request message for Documents.ImportDocuments.

  Fields:
    documentTemplate: Required. Document template used for importing all the
      documents.
    gcsSource: Optional. The Google Cloud Storage location for the documents.
      The path can include a wildcard. These URIs may have the forms `gs:///`.
      `gs:////*.`.
    importGcsCustomMetadata: Whether to import custom metadata from Google
      Cloud Storage. Only valid when the document source is Google Cloud
      Storage URI.
  """
    documentTemplate = _messages.MessageField('GoogleCloudDialogflowV2ImportDocumentTemplate', 1)
    gcsSource = _messages.MessageField('GoogleCloudDialogflowV2GcsSources', 2)
    importGcsCustomMetadata = _messages.BooleanField(3)