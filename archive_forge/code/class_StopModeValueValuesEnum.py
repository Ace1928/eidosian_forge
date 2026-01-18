from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopModeValueValuesEnum(_messages.Enum):
    """Stop modes for the existent job.

    Values:
      STOP_MODE_UNSPECIFIED: Stop mode is unspecified.
      STOP_MODE_DRAIN: Drain the job.
      STOP_MODE_CANCEL: Cancel the job.
    """
    STOP_MODE_UNSPECIFIED = 0
    STOP_MODE_DRAIN = 1
    STOP_MODE_CANCEL = 2