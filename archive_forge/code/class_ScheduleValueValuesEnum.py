from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduleValueValuesEnum(_messages.Enum):
    """The schedule for the upgrade.

    Values:
      SCHEDULE_UNSPECIFIED: Unspecified. The default is to upgrade the cluster
        immediately which is the only option today.
      IMMEDIATELY: The cluster is going to be upgraded immediately after
        receiving the request.
    """
    SCHEDULE_UNSPECIFIED = 0
    IMMEDIATELY = 1