from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1BatchProcessMetadata(_messages.Message):
    """The long-running operation metadata for BatchProcessDocuments.

  Enums:
    StateValueValuesEnum: The state of the current batch processing.

  Fields:
    createTime: The creation time of the operation.
    individualProcessStatuses: The list of response details of each document.
    state: The state of the current batch processing.
    stateMessage: A message providing more details about the current state of
      processing. For example, the error message if the operation is failed.
    updateTime: The last update time of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the current batch processing.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      WAITING: Request operation is waiting for scheduling.
      RUNNING: Request is being processed.
      SUCCEEDED: The batch processing completed successfully.
      CANCELLING: The batch processing was being cancelled.
      CANCELLED: The batch processing was cancelled.
      FAILED: The batch processing has failed.
    """
        STATE_UNSPECIFIED = 0
        WAITING = 1
        RUNNING = 2
        SUCCEEDED = 3
        CANCELLING = 4
        CANCELLED = 5
        FAILED = 6
    createTime = _messages.StringField(1)
    individualProcessStatuses = _messages.MessageField('GoogleCloudDocumentaiV1BatchProcessMetadataIndividualProcessStatus', 2, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    stateMessage = _messages.StringField(4)
    updateTime = _messages.StringField(5)