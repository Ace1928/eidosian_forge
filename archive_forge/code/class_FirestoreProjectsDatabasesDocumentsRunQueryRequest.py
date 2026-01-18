from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsRunQueryRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsRunQueryRequest object.

  Fields:
    parent: Required. The parent resource name. In the format:
      `projects/{project_id}/databases/{database_id}/documents` or `projects/{
      project_id}/databases/{database_id}/documents/{document_path}`. For
      example: `projects/my-project/databases/my-database/documents` or
      `projects/my-project/databases/my-database/documents/chatrooms/my-
      chatroom`
    runQueryRequest: A RunQueryRequest resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    runQueryRequest = _messages.MessageField('RunQueryRequest', 2)