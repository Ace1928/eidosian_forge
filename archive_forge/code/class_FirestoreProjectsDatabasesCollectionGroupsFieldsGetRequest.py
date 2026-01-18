from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCollectionGroupsFieldsGetRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCollectionGroupsFieldsGetRequest object.

  Fields:
    name: Required. A name of the form `projects/{project_id}/databases/{datab
      ase_id}/collectionGroups/{collection_id}/fields/{field_id}`
  """
    name = _messages.StringField(1, required=True)