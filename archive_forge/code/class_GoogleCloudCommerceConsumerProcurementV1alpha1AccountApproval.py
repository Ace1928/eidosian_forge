from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1AccountApproval(_messages.Message):
    """An approval for some action on an account.

  Enums:
    StateValueValuesEnum: The state of the approval.

  Fields:
    name: The name of the approval.
    reason: An explanation for the state of the approval.
    state: The state of the approval.
    updateTime: The last update timestamp of the approval.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the approval.

    Values:
      STATE_UNSPECIFIED: Sentinel value; do not use.
      PENDING: The approval is pending response from the provider. The
        approval state can transition to Account.Approval.State.APPROVED or
        Account.Approval.State.REJECTED.
      APPROVED: The approval has been granted by the provider.
      REJECTED: The approval has been rejected by the provider. A provider may
        choose to approve a previously rejected approval, so is it possible to
        transition to Account.Approval.State.APPROVED.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        APPROVED = 2
        REJECTED = 3
    name = _messages.StringField(1)
    reason = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    updateTime = _messages.StringField(4)