from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ExportDocumentRequest(_messages.Message):
    """Request message for Documents.ExportDocument.

  Fields:
    exportFullContent: When enabled, export the full content of the document
      including empirical probability.
    gcsDestination: Cloud Storage file path to export the document.
    smartMessagingPartialUpdate: When enabled, export the smart messaging
      allowlist document for partial update.
  """
    exportFullContent = _messages.BooleanField(1)
    gcsDestination = _messages.MessageField('GoogleCloudDialogflowV2GcsDestination', 2)
    smartMessagingPartialUpdate = _messages.BooleanField(3)