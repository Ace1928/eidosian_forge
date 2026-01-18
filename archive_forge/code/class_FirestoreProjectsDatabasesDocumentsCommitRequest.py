from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsCommitRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsCommitRequest object.

  Fields:
    commitRequest: A CommitRequest resource to be passed as the request body.
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`.
  """
    commitRequest = _messages.MessageField('CommitRequest', 1)
    database = _messages.StringField(2, required=True)