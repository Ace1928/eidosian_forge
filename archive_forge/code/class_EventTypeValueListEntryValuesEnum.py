from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventTypeValueListEntryValuesEnum(_messages.Enum):
    """EventTypeValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Not set, will be ignored.
      UPGRADE_AVAILABLE_EVENT: Corresponds with UpgradeAvailableEvent.
      UPGRADE_EVENT: Corresponds with UpgradeEvent.
      SECURITY_BULLETIN_EVENT: Corresponds with SecurityBulletinEvent.
    """
    EVENT_TYPE_UNSPECIFIED = 0
    UPGRADE_AVAILABLE_EVENT = 1
    UPGRADE_EVENT = 2
    SECURITY_BULLETIN_EVENT = 3