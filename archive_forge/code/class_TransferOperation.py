from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferOperation(_messages.Message):
    """A description of the execution of a transfer.

  Enums:
    StatusValueValuesEnum: Status of the transfer operation.

  Fields:
    counters: Information about the progress of the transfer operation.
    endTime: End time of this transfer execution.
    errorBreakdowns: Summarizes errors encountered with sample error log
      entries.
    loggingConfig: Cloud Logging configuration.
    name: A globally unique ID assigned by the system.
    notificationConfig: Notification configuration.
    projectId: The ID of the Google Cloud project that owns the operation.
    startTime: Start time of this transfer execution.
    status: Status of the transfer operation.
    transferJobName: The name of the transfer job that triggers this transfer
      operation.
    transferSpec: Transfer specification.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Status of the transfer operation.

    Values:
      STATUS_UNSPECIFIED: Zero is an illegal value.
      IN_PROGRESS: In progress.
      PAUSED: Paused.
      SUCCESS: Completed successfully.
      FAILED: Terminated due to an unrecoverable failure.
      ABORTED: Aborted by the user.
      QUEUED: Temporarily delayed by the system. No user action is required.
      SUSPENDING: The operation is suspending and draining the ongoing work to
        completion.
    """
        STATUS_UNSPECIFIED = 0
        IN_PROGRESS = 1
        PAUSED = 2
        SUCCESS = 3
        FAILED = 4
        ABORTED = 5
        QUEUED = 6
        SUSPENDING = 7
    counters = _messages.MessageField('TransferCounters', 1)
    endTime = _messages.StringField(2)
    errorBreakdowns = _messages.MessageField('ErrorSummary', 3, repeated=True)
    loggingConfig = _messages.MessageField('LoggingConfig', 4)
    name = _messages.StringField(5)
    notificationConfig = _messages.MessageField('NotificationConfig', 6)
    projectId = _messages.StringField(7)
    startTime = _messages.StringField(8)
    status = _messages.EnumField('StatusValueValuesEnum', 9)
    transferJobName = _messages.StringField(10)
    transferSpec = _messages.MessageField('TransferSpec', 11)