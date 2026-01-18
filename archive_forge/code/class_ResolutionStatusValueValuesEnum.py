from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResolutionStatusValueValuesEnum(_messages.Enum):
    """Error group's resolution status. An unspecified resolution status will
    be interpreted as OPEN

    Values:
      RESOLUTION_STATUS_UNSPECIFIED: Status is unknown. When left unspecified
        in requests, it is treated like OPEN.
      OPEN: The error group is not being addressed. This is the default for
        new groups. It is also used for errors re-occurring after marked
        RESOLVED.
      ACKNOWLEDGED: Error Group manually acknowledged, it can have an issue
        link attached.
      RESOLVED: Error Group manually resolved, more events for this group are
        not expected to occur.
      MUTED: The error group is muted and excluded by default on group stats
        requests.
    """
    RESOLUTION_STATUS_UNSPECIFIED = 0
    OPEN = 1
    ACKNOWLEDGED = 2
    RESOLVED = 3
    MUTED = 4