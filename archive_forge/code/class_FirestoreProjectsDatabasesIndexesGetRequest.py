from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesIndexesGetRequest(_messages.Message):
    """A FirestoreProjectsDatabasesIndexesGetRequest object.

  Fields:
    name: The name of the index. For example:
      `projects/{project_id}/databases/{database_id}/indexes/{index_id}`
  """
    name = _messages.StringField(1, required=True)