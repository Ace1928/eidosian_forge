from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppendFsyncValueValuesEnum(_messages.Enum):
    """Optional. fsync configuration.

    Values:
      APPEND_FSYNC_UNSPECIFIED: Not set. Default: EVERYSEC
      NO: Never fsync. Normally Linux will flush data every 30 seconds with
        this configuration, but it's up to the kernel's exact tuning.
      EVERYSEC: fsync every second. Fast enough, and you may lose 1 second of
        data if there is a disaster
      ALWAYS: fsync every time new commands are appended to the AOF. It has
        the best data loss protection at the cost of performance
    """
    APPEND_FSYNC_UNSPECIFIED = 0
    NO = 1
    EVERYSEC = 2
    ALWAYS = 3