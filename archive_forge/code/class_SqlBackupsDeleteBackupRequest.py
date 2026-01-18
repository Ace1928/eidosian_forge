from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupsDeleteBackupRequest(_messages.Message):
    """A SqlBackupsDeleteBackupRequest object.

  Fields:
    name: Required. The name of the backup to delete. Format:
      projects/{project}/backups/{backup}
  """
    name = _messages.StringField(1, required=True)