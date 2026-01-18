from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreTableMetadata(_messages.Message):
    """Metadata type for the long-running operation returned by RestoreTable.

  Enums:
    SourceTypeValueValuesEnum: The type of the restore source.

  Fields:
    backupInfo: A BackupInfo attribute.
    name: Name of the table being created and restored to.
    optimizeTableOperationName: If exists, the name of the long-running
      operation that will be used to track the post-restore optimization
      process to optimize the performance of the restored table. The metadata
      type of the long-running operation is OptimizeRestoreTableMetadata. The
      response type is Empty. This long-running operation may be automatically
      created by the system if applicable after the RestoreTable long-running
      operation completes successfully. This operation may not be created if
      the table is already optimized or the restore was not successful.
    progress: The progress of the RestoreTable operation.
    sourceType: The type of the restore source.
  """

    class SourceTypeValueValuesEnum(_messages.Enum):
        """The type of the restore source.

    Values:
      RESTORE_SOURCE_TYPE_UNSPECIFIED: No restore associated.
      BACKUP: A backup was used as the source of the restore.
    """
        RESTORE_SOURCE_TYPE_UNSPECIFIED = 0
        BACKUP = 1
    backupInfo = _messages.MessageField('BackupInfo', 1)
    name = _messages.StringField(2)
    optimizeTableOperationName = _messages.StringField(3)
    progress = _messages.MessageField('OperationProgress', 4)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 5)