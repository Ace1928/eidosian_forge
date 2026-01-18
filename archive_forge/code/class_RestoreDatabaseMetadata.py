from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreDatabaseMetadata(_messages.Message):
    """Metadata type for the long-running operation returned by
  RestoreDatabase.

  Enums:
    SourceTypeValueValuesEnum: The type of the restore source.

  Fields:
    backupInfo: Information about the backup used to restore the database.
    cancelTime: The time at which cancellation of this operation was received.
      Operations.CancelOperation starts asynchronous cancellation on a long-
      running operation. The server makes a best effort to cancel the
      operation, but success is not guaranteed. Clients can use
      Operations.GetOperation or other methods to check whether the
      cancellation succeeded or whether the operation completed despite
      cancellation. On successful cancellation, the operation is not deleted;
      instead, it becomes an operation with an Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    name: Name of the database being created and restored to.
    optimizeDatabaseOperationName: If exists, the name of the long-running
      operation that will be used to track the post-restore optimization
      process to optimize the performance of the restored database, and remove
      the dependency on the restore source. The name is of the form
      `projects//instances//databases//operations/` where the is the name of
      database being created and restored to. The metadata type of the long-
      running operation is OptimizeRestoredDatabaseMetadata. This long-running
      operation will be automatically created by the system after the
      RestoreDatabase long-running operation completes successfully. This
      operation will not be created if the restore was not successful.
    progress: The progress of the RestoreDatabase operation.
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
    cancelTime = _messages.StringField(2)
    name = _messages.StringField(3)
    optimizeDatabaseOperationName = _messages.StringField(4)
    progress = _messages.MessageField('OperationProgress', 5)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 6)