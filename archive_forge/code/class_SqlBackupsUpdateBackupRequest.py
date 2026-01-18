from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupsUpdateBackupRequest(_messages.Message):
    """A SqlBackupsUpdateBackupRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: Output only. The resource name of the backup. Format:
      projects/{project}/backups/{backup}
    updateMask: The list of fields to update. Only final backup retention
      period and description can be updated.
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)