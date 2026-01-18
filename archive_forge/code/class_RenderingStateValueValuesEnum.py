from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenderingStateValueValuesEnum(_messages.Enum):
    """Output only. Current state of the render operation for this Target.

    Values:
      TARGET_RENDER_STATE_UNSPECIFIED: The render operation state is
        unspecified.
      SUCCEEDED: The render operation has completed successfully.
      FAILED: The render operation has failed.
      IN_PROGRESS: The render operation is in progress.
    """
    TARGET_RENDER_STATE_UNSPECIFIED = 0
    SUCCEEDED = 1
    FAILED = 2
    IN_PROGRESS = 3