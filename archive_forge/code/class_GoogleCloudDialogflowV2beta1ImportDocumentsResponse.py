from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1ImportDocumentsResponse(_messages.Message):
    """Response message for Documents.ImportDocuments.

  Fields:
    warnings: Includes details about skipped documents or any other warnings.
  """
    warnings = _messages.MessageField('GoogleRpcStatus', 1, repeated=True)