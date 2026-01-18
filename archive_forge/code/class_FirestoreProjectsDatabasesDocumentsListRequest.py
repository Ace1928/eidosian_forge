from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsListRequest object.

  Fields:
    collectionId: Optional. The collection ID, relative to `parent`, to list.
      For example: `chatrooms` or `messages`. This is optional, and when not
      provided, Firestore will list documents from all collections under the
      provided `parent`.
    mask_fieldPaths: The list of field paths in the mask. See Document.fields
      for a field path syntax reference.
    orderBy: Optional. The optional ordering of the documents to return. For
      example: `priority desc, __name__ desc`. This mirrors the `ORDER BY`
      used in Firestore queries but in a string representation. When absent,
      documents are ordered based on `__name__ ASC`.
    pageSize: Optional. The maximum number of documents to return in a single
      response. Firestore may return fewer than this value.
    pageToken: Optional. A page token, received from a previous
      `ListDocuments` response. Provide this to retrieve the subsequent page.
      When paginating, all other parameters (with the exception of
      `page_size`) must match the values set in the request that generated the
      page token.
    parent: Required. The parent resource name. In the format:
      `projects/{project_id}/databases/{database_id}/documents` or `projects/{
      project_id}/databases/{database_id}/documents/{document_path}`. For
      example: `projects/my-project/databases/my-database/documents` or
      `projects/my-project/databases/my-database/documents/chatrooms/my-
      chatroom`
    readTime: Perform the read at the provided time. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    showMissing: If the list should show missing documents. A document is
      missing if it does not exist, but there are sub-documents nested
      underneath it. When true, such missing documents will be returned with a
      key but will not have fields, `create_time`, or `update_time` set.
      Requests with `show_missing` may not specify `where` or `order_by`.
    transaction: Perform the read as part of an already active transaction.
  """
    collectionId = _messages.StringField(1, required=True)
    mask_fieldPaths = _messages.StringField(2, repeated=True)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)
    readTime = _messages.StringField(7)
    showMissing = _messages.BooleanField(8)
    transaction = _messages.BytesField(9)