from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SyncModeValueValuesEnum(_messages.Enum):
    """External sync mode

    Values:
      EXTERNAL_SYNC_MODE_UNSPECIFIED: Unknown external sync mode, will be
        defaulted to ONLINE mode
      ONLINE: Online external sync will set up replication after initial data
        external sync
      OFFLINE: Offline external sync only dumps and loads a one-time snapshot
        of the primary instance's data
    """
    EXTERNAL_SYNC_MODE_UNSPECIFIED = 0
    ONLINE = 1
    OFFLINE = 2