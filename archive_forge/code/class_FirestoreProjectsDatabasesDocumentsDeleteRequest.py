from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsDeleteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsDeleteRequest object.

  Fields:
    currentDocument_exists: When set to `true`, the target document must
      exist. When set to `false`, the target document must not exist.
    currentDocument_updateTime: When set, the target document must exist and
      have been last updated at that time. Timestamp must be microsecond
      aligned.
    name: Required. The resource name of the Document to delete. In the
      format: `projects/{project_id}/databases/{database_id}/documents/{docume
      nt_path}`.
  """
    currentDocument_exists = _messages.BooleanField(1)
    currentDocument_updateTime = _messages.StringField(2)
    name = _messages.StringField(3, required=True)