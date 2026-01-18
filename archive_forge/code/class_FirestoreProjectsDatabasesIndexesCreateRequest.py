from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesIndexesCreateRequest(_messages.Message):
    """A FirestoreProjectsDatabasesIndexesCreateRequest object.

  Fields:
    googleFirestoreAdminV1beta1Index: A GoogleFirestoreAdminV1beta1Index
      resource to be passed as the request body.
    parent: The name of the database this index will apply to. For example:
      `projects/{project_id}/databases/{database_id}`
  """
    googleFirestoreAdminV1beta1Index = _messages.MessageField('GoogleFirestoreAdminV1beta1Index', 1)
    parent = _messages.StringField(2, required=True)