from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RestoreBackupContext(_messages.Message):
    """Database instance restore from backup context. Backup context contains
  source instance id and project id.

  Fields:
    backupRunId: The ID of the backup run to restore from.
    instanceId: The ID of the instance that the backup was taken from.
    kind: This is always `sql#restoreBackupContext`.
    project: The full project ID of the source instance.
  """
    backupRunId = _messages.IntegerField(1)
    instanceId = _messages.StringField(2)
    kind = _messages.StringField(3)
    project = _messages.StringField(4)