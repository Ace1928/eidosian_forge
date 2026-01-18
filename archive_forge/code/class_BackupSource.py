from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupSource(_messages.Message):
    """Message describing a BackupSource.

  Fields:
    backupName: Required. The name of the backup resource with the format: *
      projects/{project}/locations/{region}/backups/{backup_id}
    backupUid: Output only. The system-generated UID of the backup which was
      used to create this resource. The UID is generated when the backup is
      created, and it is retained until the backup is deleted.
  """
    backupName = _messages.StringField(1)
    backupUid = _messages.StringField(2)