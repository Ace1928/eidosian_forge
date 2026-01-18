from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementStateValueValuesEnum(_messages.Enum):
    """Output only. Management state of the user on the device.

    Values:
      MANAGEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      WIPING: This user's data and profile is being removed from the device.
      WIPED: This user's data and profile is removed from the device.
      APPROVED: User is approved to access data on the device.
      BLOCKED: User is blocked from accessing data on the device.
      PENDING_APPROVAL: User is awaiting approval.
      UNENROLLED: User is unenrolled from Advanced Windows Management, but the
        Windows account is still intact.
    """
    MANAGEMENT_STATE_UNSPECIFIED = 0
    WIPING = 1
    WIPED = 2
    APPROVED = 3
    BLOCKED = 4
    PENDING_APPROVAL = 5
    UNENROLLED = 6