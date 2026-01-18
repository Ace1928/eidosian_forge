from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesRestoreRequest(_messages.Message):
    """A FirestoreProjectsDatabasesRestoreRequest object.

  Fields:
    googleFirestoreAdminV1RestoreDatabaseRequest: A
      GoogleFirestoreAdminV1RestoreDatabaseRequest resource to be passed as
      the request body.
    parent: Required. The project to restore the database in. Format is
      `projects/{project_id}`.
  """
    googleFirestoreAdminV1RestoreDatabaseRequest = _messages.MessageField('GoogleFirestoreAdminV1RestoreDatabaseRequest', 1)
    parent = _messages.StringField(2, required=True)