from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityCustomersUserinvitationsSendRequest(_messages.Message):
    """A CloudidentityCustomersUserinvitationsSendRequest object.

  Fields:
    name: Required. `UserInvitation` name in the format
      `customers/{customer}/userinvitations/{user_email_address}`
    sendUserInvitationRequest: A SendUserInvitationRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    sendUserInvitationRequest = _messages.MessageField('SendUserInvitationRequest', 2)