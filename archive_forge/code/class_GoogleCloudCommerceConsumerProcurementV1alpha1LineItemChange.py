from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1LineItemChange(_messages.Message):
    """A change made on a line item.

  Enums:
    ChangeStateValueValuesEnum: Output only. State of the change.
    ChangeStateReasonTypeValueValuesEnum: Output only. Predefined enum types
      for why this line item change is in current state. For example, a line
      item change's state could be `LINE_ITEM_CHANGE_STATE_COMPLETED` because
      of end-of-term expiration, immediate cancellation initiated by the user,
      or system-initiated cancellation.
    ChangeTypeValueValuesEnum: Required. Type of the change to make.

  Fields:
    changeEffectiveTime: Output only. A time at which the change became or
      will become (in case of pending change) effective.
    changeId: Output only. Change ID. All changes made within one order update
      operation have the same change_id.
    changeState: Output only. State of the change.
    changeStateReasonType: Output only. Predefined enum types for why this
      line item change is in current state. For example, a line item change's
      state could be `LINE_ITEM_CHANGE_STATE_COMPLETED` because of end-of-term
      expiration, immediate cancellation initiated by the user, or system-
      initiated cancellation.
    changeType: Required. Type of the change to make.
    createTime: Output only. The time when change was initiated.
    newLineItemInfo: Line item info after the change.
    oldLineItemInfo: Output only. Line item info before the change.
    stateReason: Output only. Provider-supplied message explaining the
      LineItemChange's state. Mainly used to communicate progress and ETA for
      provisioning in the case of `PENDING_APPROVAL`, and to explain why the
      change request was denied or canceled in the case of `REJECTED` and
      `CANCELED` states.
    updateTime: Output only. The time when change was updated, e.g.
      approved/rejected by partners or cancelled by the user.
  """

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

    class ChangeStateValueValuesEnum(_messages.Enum):
        """Output only. State of the change.

    Values:
      LINE_ITEM_CHANGE_STATE_UNSPECIFIED: Sentinel value. Do not use.
      LINE_ITEM_CHANGE_STATE_PENDING_APPROVAL: Change is in this state when a
        change is initiated and waiting for partner approval. This state is
        only applicable for pending change.
      LINE_ITEM_CHANGE_STATE_APPROVED: Change is in this state after it's
        approved by the partner or auto-approved but before it takes effect.
        The change can be overwritten or cancelled depending on the new line
        item info property (pending Private Offer change cannot be cancelled
        and can only be overwritten by another Private Offer). This state is
        only applicable for pending change.
      LINE_ITEM_CHANGE_STATE_COMPLETED: Change is in this state after it's
        been activated. This state is only applicable for change in history.
      LINE_ITEM_CHANGE_STATE_REJECTED: Change is in this state if it was
        rejected by the partner. This state is only applicable for change in
        history.
      LINE_ITEM_CHANGE_STATE_ABANDONED: Change is in this state if it was
        abandoned by the user. This state is only applicable for change in
        history.
      LINE_ITEM_CHANGE_STATE_ACTIVATING: Change is in this state if it's
        currently being provisioned downstream. The change can't be
        overwritten or cancelled when it's in this state. This state is only
        applicable for pending change.
    """
        LINE_ITEM_CHANGE_STATE_UNSPECIFIED = 0
        LINE_ITEM_CHANGE_STATE_PENDING_APPROVAL = 1
        LINE_ITEM_CHANGE_STATE_APPROVED = 2
        LINE_ITEM_CHANGE_STATE_COMPLETED = 3
        LINE_ITEM_CHANGE_STATE_REJECTED = 4
        LINE_ITEM_CHANGE_STATE_ABANDONED = 5
        LINE_ITEM_CHANGE_STATE_ACTIVATING = 6

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """Required. Type of the change to make.

    Values:
      LINE_ITEM_CHANGE_TYPE_UNSPECIFIED: Sentinel value. Do not use.
      LINE_ITEM_CHANGE_TYPE_CREATE: The change is to create a new line item.
      LINE_ITEM_CHANGE_TYPE_UPDATE: The change is to update an existing line
        item.
      LINE_ITEM_CHANGE_TYPE_CANCEL: The change is to cancel an existing line
        item.
      LINE_ITEM_CHANGE_TYPE_REVERT_CANCELLATION: The change is to revert a
        cancellation.
    """
        LINE_ITEM_CHANGE_TYPE_UNSPECIFIED = 0
        LINE_ITEM_CHANGE_TYPE_CREATE = 1
        LINE_ITEM_CHANGE_TYPE_UPDATE = 2
        LINE_ITEM_CHANGE_TYPE_CANCEL = 3
        LINE_ITEM_CHANGE_TYPE_REVERT_CANCELLATION = 4
    changeEffectiveTime = _messages.StringField(1)
    changeId = _messages.StringField(2)
    changeState = _messages.EnumField('ChangeStateValueValuesEnum', 3)
    changeStateReasonType = _messages.EnumField('ChangeStateReasonTypeValueValuesEnum', 4)
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 5)
    createTime = _messages.StringField(6)
    newLineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 7)
    oldLineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 8)
    stateReason = _messages.StringField(9)
    updateTime = _messages.StringField(10)