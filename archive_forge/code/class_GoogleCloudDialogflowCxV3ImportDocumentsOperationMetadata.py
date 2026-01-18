from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportDocumentsOperationMetadata(_messages.Message):
    """Metadata for ImportDocuments operation.

  Fields:
    genericMetadata: The generic information of the operation.
  """
    genericMetadata = _messages.MessageField('GoogleCloudDialogflowCxV3GenericKnowledgeOperationMetadata', 1)