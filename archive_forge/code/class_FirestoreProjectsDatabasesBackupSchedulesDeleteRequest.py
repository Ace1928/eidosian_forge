from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesBackupSchedulesDeleteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesBackupSchedulesDeleteRequest object.

  Fields:
    name: Required. The name of the backup schedule. Format `projects/{project
      }/databases/{database}/backupSchedules/{backup_schedule}`
  """
    name = _messages.StringField(1, required=True)