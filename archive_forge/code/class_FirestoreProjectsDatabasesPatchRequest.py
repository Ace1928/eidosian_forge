from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesPatchRequest(_messages.Message):
    """A FirestoreProjectsDatabasesPatchRequest object.

  Fields:
    googleFirestoreAdminV1Database: A GoogleFirestoreAdminV1Database resource
      to be passed as the request body.
    name: The resource name of the Database. Format:
      `projects/{project}/databases/{database}`
    updateMask: The list of fields to be updated.
  """
    googleFirestoreAdminV1Database = _messages.MessageField('GoogleFirestoreAdminV1Database', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)