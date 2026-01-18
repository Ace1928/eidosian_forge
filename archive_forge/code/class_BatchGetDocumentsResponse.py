from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetDocumentsResponse(_messages.Message):
    """The streamed response for Firestore.BatchGetDocuments.

  Fields:
    found: A document that was requested.
    missing: A document name that was requested but does not exist. In the
      format: `projects/{project_id}/databases/{database_id}/documents/{docume
      nt_path}`.
    readTime: The time at which the document was read. This may be monotically
      increasing, in this case the previous documents in the result stream are
      guaranteed not to have changed between their read_time and this one.
    transaction: The transaction that was started as part of this request.
      Will only be set in the first response, and only if
      BatchGetDocumentsRequest.new_transaction was set in the request.
  """
    found = _messages.MessageField('Document', 1)
    missing = _messages.StringField(2)
    readTime = _messages.StringField(3)
    transaction = _messages.BytesField(4)