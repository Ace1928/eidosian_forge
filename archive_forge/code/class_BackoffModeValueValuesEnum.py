from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackoffModeValueValuesEnum(_messages.Enum):
    """Output only. The pattern of how the wait time of the retry attempt is
    calculated.

    Values:
      BACKOFF_MODE_UNSPECIFIED: No WaitMode is specified.
      BACKOFF_MODE_LINEAR: Increases the wait time linearly.
      BACKOFF_MODE_EXPONENTIAL: Increases the wait time exponentially.
    """
    BACKOFF_MODE_UNSPECIFIED = 0
    BACKOFF_MODE_LINEAR = 1
    BACKOFF_MODE_EXPONENTIAL = 2