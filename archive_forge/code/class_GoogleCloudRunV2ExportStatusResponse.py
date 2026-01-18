from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ExportStatusResponse(_messages.Message):
    """ExportStatusResponse contains the status of image export operation, with
  the status of each image export job.

  Enums:
    OperationStateValueValuesEnum: Output only. The state of the overall
      export operation.

  Fields:
    imageExportStatuses: The status of each image export job.
    operationId: The operation id.
    operationState: Output only. The state of the overall export operation.
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """Output only. The state of the overall export operation.

    Values:
      OPERATION_STATE_UNSPECIFIED: State unspecified.
      IN_PROGRESS: Operation still in progress.
      FINISHED: Operation finished.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        FINISHED = 2
    imageExportStatuses = _messages.MessageField('GoogleCloudRunV2ImageExportStatus', 1, repeated=True)
    operationId = _messages.StringField(2)
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 3)