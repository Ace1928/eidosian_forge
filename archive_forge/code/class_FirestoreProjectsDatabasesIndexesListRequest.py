from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesIndexesListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesIndexesListRequest object.

  Fields:
    filter: A string attribute.
    pageSize: The standard List page size.
    pageToken: The standard List page token.
    parent: The database name. For example:
      `projects/{project_id}/databases/{database_id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)