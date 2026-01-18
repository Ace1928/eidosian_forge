from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsPatchRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsPatchRequest object.

  Fields:
    currentDocument_exists: When set to `true`, the target document must
      exist. When set to `false`, the target document must not exist.
    currentDocument_updateTime: When set, the target document must exist and
      have been last updated at that time. Timestamp must be microsecond
      aligned.
    document: A Document resource to be passed as the request body.
    mask_fieldPaths: The list of field paths in the mask. See Document.fields
      for a field path syntax reference.
    name: The resource name of the document, for example
      `projects/{project_id}/databases/{database_id}/documents/{document_path}
      `.
    updateMask_fieldPaths: The list of field paths in the mask. See
      Document.fields for a field path syntax reference.
  """
    currentDocument_exists = _messages.BooleanField(1)
    currentDocument_updateTime = _messages.StringField(2)
    document = _messages.MessageField('Document', 3)
    mask_fieldPaths = _messages.StringField(4, repeated=True)
    name = _messages.StringField(5, required=True)
    updateMask_fieldPaths = _messages.StringField(6, repeated=True)