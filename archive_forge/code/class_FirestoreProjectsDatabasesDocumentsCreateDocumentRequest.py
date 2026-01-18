from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsCreateDocumentRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsCreateDocumentRequest object.

  Fields:
    collectionId: Required. The collection ID, relative to `parent`, to list.
      For example: `chatrooms`.
    document: A Document resource to be passed as the request body.
    documentId: The client-assigned document ID to use for this document.
      Optional. If not specified, an ID will be assigned by the service.
    mask_fieldPaths: The list of field paths in the mask. See Document.fields
      for a field path syntax reference.
    parent: Required. The parent resource. For example:
      `projects/{project_id}/databases/{database_id}/documents` or `projects/{
      project_id}/databases/{database_id}/documents/chatrooms/{chatroom_id}`
  """
    collectionId = _messages.StringField(1, required=True)
    document = _messages.MessageField('Document', 2)
    documentId = _messages.StringField(3)
    mask_fieldPaths = _messages.StringField(4, repeated=True)
    parent = _messages.StringField(5, required=True)