from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesBackupSchedulesListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesBackupSchedulesListRequest object.

  Fields:
    parent: Required. The parent database. Format is
      `projects/{project}/databases/{database}`.
  """
    parent = _messages.StringField(1, required=True)