from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1GcsDocuments(_messages.Message):
    """Specifies a set of documents on Cloud Storage.

  Fields:
    documents: The list of documents.
  """
    documents = _messages.MessageField('GoogleCloudDocumentaiV1GcsDocument', 1, repeated=True)