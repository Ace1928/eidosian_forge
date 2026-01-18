from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDeleteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDeleteRequest object.

  Fields:
    etag: The current etag of the Database. If an etag is provided and does
      not match the current etag of the database, deletion will be blocked and
      a FAILED_PRECONDITION error will be returned.
    name: Required. A name of the form
      `projects/{project_id}/databases/{database_id}`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)