from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreParameters(_messages.Message):
    """The RestoreParameters if volume is created from a snapshot or backup.

  Fields:
    sourceBackup: Full name of the backup resource. Format: projects/{project}
      /locations/{location}/backupVaults/{backup_vault_id}/backups/{backup_id}
    sourceSnapshot: Full name of the snapshot resource. Format: projects/{proj
      ect}/locations/{location}/volumes/{volume}/snapshots/{snapshot}
  """
    sourceBackup = _messages.StringField(1)
    sourceSnapshot = _messages.StringField(2)