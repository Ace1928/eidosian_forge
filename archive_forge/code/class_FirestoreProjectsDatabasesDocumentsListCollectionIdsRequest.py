from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsListCollectionIdsRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsListCollectionIdsRequest object.

  Fields:
    listCollectionIdsRequest: A ListCollectionIdsRequest resource to be passed
      as the request body.
    parent: Required. The parent document. In the format:
      `projects/{project_id}/databases/{database_id}/documents/{document_path}
      `. For example: `projects/my-project/databases/my-
      database/documents/chatrooms/my-chatroom`
  """
    listCollectionIdsRequest = _messages.MessageField('ListCollectionIdsRequest', 1)
    parent = _messages.StringField(2, required=True)