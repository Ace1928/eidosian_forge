from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCollectionGroupsIndexesDeleteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCollectionGroupsIndexesDeleteRequest object.

  Fields:
    name: Required. A name of the form `projects/{project_id}/databases/{datab
      ase_id}/collectionGroups/{collection_id}/indexes/{index_id}`
  """
    name = _messages.StringField(1, required=True)