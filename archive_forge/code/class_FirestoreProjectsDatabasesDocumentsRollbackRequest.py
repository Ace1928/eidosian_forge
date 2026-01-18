from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsRollbackRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsRollbackRequest object.

  Fields:
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`.
    rollbackRequest: A RollbackRequest resource to be passed as the request
      body.
  """
    database = _messages.StringField(1, required=True)
    rollbackRequest = _messages.MessageField('RollbackRequest', 2)