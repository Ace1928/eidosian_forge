from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1EntitlementChange(_messages.Message):
    """Entitlement change information. Next Id: 8

  Enums:
    ChangeStateValueValuesEnum: Output only. State of the change.
    ChangeStateReasonTypeValueValuesEnum: Output only. Predefined enum types
      for why this change is in current state.

  Fields:
    changeEffectiveTime: Output only. A time at which the change became or
      will become (in case of pending change) effective.
    changeState: Output only. State of the change.
    changeStateReasonType: Output only. Predefined enum types for why this
      change is in current state.
    newFlavorExternalName: Output only. Flavor external name after the change.
    oldFlavorExternalName: Output only. Flavor external name before the
      change.
  """

    class ChangeStateReasonTypeValueValuesEnum(_messages.Enum):
        """Output only. Predefined enum types for why this change is in current
    state.

    Values:
      CHANGE_STATE_REASON_TYPE_UNSPECIFIED: Default value, indicating there is
        no predefined type for change state reason.
      CHANGE_STATE_REASON_TYPE_EXPIRED: Change is in current state due to term
        expiration.
      CHANGE_STATE_REASON_TYPE_USER_CANCELLED: Change is in current state due
        to user explicit cancellation.
      CHANGE_STATE_REASON_TYPE_SYSTEM_CANCELLED: Change is in current state
        due to system cancellation.
    """
        CHANGE_STATE_REASON_TYPE_UNSPECIFIED = 0
        CHANGE_STATE_REASON_TYPE_EXPIRED = 1
        CHANGE_STATE_REASON_TYPE_USER_CANCELLED = 2
        CHANGE_STATE_REASON_TYPE_SYSTEM_CANCELLED = 3

    class ChangeStateValueValuesEnum(_messages.Enum):
        """Output only. State of the change.

    Values:
      CHANGE_STATE_UNSPECIFIED: Sentinel value. Do not use.
      CHANGE_STATE_PENDING_APPROVAL: Change is in this state when a change is
        initiated and waiting for partner approval. This state is only
        applicable for pending change.
      CHANGE_STATE_APPROVED: Change is in this state, if the change was
        approved by partner or auto-approved but is pending to be effective.
        The change can be overwritten or cancelled depending on the new line
        item info property (pending Private Offer change cannot be cancelled
        and can only be overwritten by another Private Offer). This state is
        only applicable for pending change.
      CHANGE_STATE_COMPLETED: Change is in this state, if the change was
        activated and completed successfully. This state is only applicable
        for change in history.
      CHANGE_STATE_REJECTED: Change is in this state, if the change was
        rejected by partner. This state is only applicable for change in
        history.
      CHANGE_STATE_ABANDONED: Change is in this state, if it was abandoned by
        user. This state is only applicable for change in history.
      CHANGE_STATE_ACTIVATING: Change is in this state, if it is going through
        downstream provision, the change cannot be overwritten or cancelled in
        this state. This state is only applicable for pending change.
    """
        CHANGE_STATE_UNSPECIFIED = 0
        CHANGE_STATE_PENDING_APPROVAL = 1
        CHANGE_STATE_APPROVED = 2
        CHANGE_STATE_COMPLETED = 3
        CHANGE_STATE_REJECTED = 4
        CHANGE_STATE_ABANDONED = 5
        CHANGE_STATE_ACTIVATING = 6
    changeEffectiveTime = _messages.StringField(1)
    changeState = _messages.EnumField('ChangeStateValueValuesEnum', 2)
    changeStateReasonType = _messages.EnumField('ChangeStateReasonTypeValueValuesEnum', 3)
    newFlavorExternalName = _messages.StringField(4)
    oldFlavorExternalName = _messages.StringField(5)