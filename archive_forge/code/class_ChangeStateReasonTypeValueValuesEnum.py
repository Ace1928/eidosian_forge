from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeStateReasonTypeValueValuesEnum(_messages.Enum):
    """Output only. Predefined enum types for why this line item change is in
    current state. For example, a line item change's state could be
    `LINE_ITEM_CHANGE_STATE_COMPLETED` because of end-of-term expiration,
    immediate cancellation initiated by the user, or system-initiated
    cancellation.

    Values:
      LINE_ITEM_CHANGE_STATE_REASON_TYPE_UNSPECIFIED: Default value,
        indicating there's no predefined type for change state reason.
      LINE_ITEM_CHANGE_STATE_REASON_TYPE_EXPIRED: Change is in current state
        due to term expiration.
      LINE_ITEM_CHANGE_STATE_REASON_TYPE_USER_CANCELLED: Change is in current
        state due to user-initiated cancellation.
      LINE_ITEM_CHANGE_STATE_REASON_TYPE_SYSTEM_CANCELLED: Change is in
        current state due to system-initiated cancellation.
    """
    LINE_ITEM_CHANGE_STATE_REASON_TYPE_UNSPECIFIED = 0
    LINE_ITEM_CHANGE_STATE_REASON_TYPE_EXPIRED = 1
    LINE_ITEM_CHANGE_STATE_REASON_TYPE_USER_CANCELLED = 2
    LINE_ITEM_CHANGE_STATE_REASON_TYPE_SYSTEM_CANCELLED = 3