from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreInfo(_messages.Message):
    """Information about the database restore.

  Enums:
    SourceTypeValueValuesEnum: The type of the restore source.

  Fields:
    backupInfo: Information about the backup used to restore the database. The
      backup may no longer exist.
    sourceType: The type of the restore source.
  """

    class SourceTypeValueValuesEnum(_messages.Enum):
        """The type of the restore source.

    Values:
      TYPE_UNSPECIFIED: No restore associated.
      BACKUP: A backup was used as the source of the restore.
    """
        TYPE_UNSPECIFIED = 0
        BACKUP = 1
    backupInfo = _messages.MessageField('BackupInfo', 1)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 2)