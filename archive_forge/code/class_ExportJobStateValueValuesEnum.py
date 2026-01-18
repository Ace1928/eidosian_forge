from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportJobStateValueValuesEnum(_messages.Enum):
    """Output only. Has the image export job finished (regardless of
    successful or failure).

    Values:
      EXPORT_JOB_STATE_UNSPECIFIED: State unspecified.
      IN_PROGRESS: Job still in progress.
      FINISHED: Job finished.
    """
    EXPORT_JOB_STATE_UNSPECIFIED = 0
    IN_PROGRESS = 1
    FINISHED = 2