from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyBackupMetadata(_messages.Message):
    """Metadata type for the operation returned by CopyBackup.

  Fields:
    cancelTime: The time at which cancellation of CopyBackup operation was
      received. Operations.CancelOperation starts asynchronous cancellation on
      a long-running operation. The server makes a best effort to cancel the
      operation, but success is not guaranteed. Clients can use
      Operations.GetOperation or other methods to check whether the
      cancellation succeeded or whether the operation completed despite
      cancellation. On successful cancellation, the operation is not deleted;
      instead, it becomes an operation with an Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    name: The name of the backup being created through the copy operation.
      Values are of the form `projects//instances//backups/`.
    progress: The progress of the CopyBackup operation.
    sourceBackup: The name of the source backup that is being copied. Values
      are of the form `projects//instances//backups/`.
  """
    cancelTime = _messages.StringField(1)
    name = _messages.StringField(2)
    progress = _messages.MessageField('OperationProgress', 3)
    sourceBackup = _messages.StringField(4)