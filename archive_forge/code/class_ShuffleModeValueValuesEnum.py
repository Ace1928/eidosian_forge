from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShuffleModeValueValuesEnum(_messages.Enum):
    """Output only. The shuffle mode used for the job.

    Values:
      SHUFFLE_MODE_UNSPECIFIED: Shuffle mode information is not available.
      VM_BASED: Shuffle is done on the worker VMs.
      SERVICE_BASED: Shuffle is done on the service side.
    """
    SHUFFLE_MODE_UNSPECIFIED = 0
    VM_BASED = 1
    SERVICE_BASED = 2