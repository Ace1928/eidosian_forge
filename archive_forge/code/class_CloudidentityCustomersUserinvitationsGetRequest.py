from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityCustomersUserinvitationsGetRequest(_messages.Message):
    """A CloudidentityCustomersUserinvitationsGetRequest object.

  Fields:
    name: Required. `UserInvitation` name in the format
      `customers/{customer}/userinvitations/{user_email_address}`
  """
    name = _messages.StringField(1, required=True)