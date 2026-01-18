from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserInvitation(_messages.Message):
    """The `UserInvitation` resource represents an email that can be sent to an
  unmanaged user account inviting them to join the customer's Google Workspace
  or Cloud Identity account. An unmanaged account shares an email address
  domain with the Google Workspace or Cloud Identity account but is not
  managed by it yet. If the user accepts the `UserInvitation`, the user
  account will become managed.

  Enums:
    StateValueValuesEnum: State of the `UserInvitation`.

  Fields:
    mailsSentCount: Number of invitation emails sent to the user.
    name: Shall be of the form
      `customers/{customer}/userinvitations/{user_email_address}`.
    state: State of the `UserInvitation`.
    updateTime: Time when the `UserInvitation` was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the `UserInvitation`.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      NOT_YET_SENT: The `UserInvitation` has been created and is ready for
        sending as an email.
      INVITED: The user has been invited by email.
      ACCEPTED: The user has accepted the invitation and is part of the
        organization.
      DECLINED: The user declined the invitation.
    """
        STATE_UNSPECIFIED = 0
        NOT_YET_SENT = 1
        INVITED = 2
        ACCEPTED = 3
        DECLINED = 4
    mailsSentCount = _messages.IntegerField(1)
    name = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    updateTime = _messages.StringField(4)