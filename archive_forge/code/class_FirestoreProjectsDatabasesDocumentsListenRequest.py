from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsListenRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsListenRequest object.

  Fields:
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`.
    listenRequest: A ListenRequest resource to be passed as the request body.
  """
    database = _messages.StringField(1, required=True)
    listenRequest = _messages.MessageField('ListenRequest', 2)