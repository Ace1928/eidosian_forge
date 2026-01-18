from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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