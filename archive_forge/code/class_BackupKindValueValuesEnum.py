from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BackupKindValueValuesEnum(_messages.Enum):
    """Specifies the kind of backup, PHYSICAL or DEFAULT_SNAPSHOT.

    Values:
      SQL_BACKUP_KIND_UNSPECIFIED: This is an unknown BackupKind.
      SNAPSHOT: The snapshot based backups
      PHYSICAL: Physical backups
    """
    SQL_BACKUP_KIND_UNSPECIFIED = 0
    SNAPSHOT = 1
    PHYSICAL = 2