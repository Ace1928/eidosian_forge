from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LastBackupStateValueValuesEnum(_messages.Enum):
    """Output only. The status of the last backup to this BackupVault

    Values:
      LAST_BACKUP_STATE_UNSPECIFIED: Status not set.
      FIRST_BACKUP_PENDING: The first backup has not yet completed
      SUCCEEDED: The most recent backup was successful
      FAILED: The most recent backup failed
      PERMISSION_DENIED: The most recent backup could not be run/failed
        because of the lack of permissions
    """
    LAST_BACKUP_STATE_UNSPECIFIED = 0
    FIRST_BACKUP_PENDING = 1
    SUCCEEDED = 2
    FAILED = 3
    PERMISSION_DENIED = 4