from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BackupContext(_messages.Message):
    """Backup context.

  Fields:
    backupId: The identifier of the backup.
    kind: This is always `sql#backupContext`.
    name: The name of the backup. Format: projects/{project}/backups/{backup}
  """
    backupId = _messages.IntegerField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3)