from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateCreateMembershipRequest(_messages.Message):
    """Request message for the `GkeHub.ValidateCreateMembership` method.

  Fields:
    membership: Required. Membership resource to be created.
    membershipId: Required. Client chosen membership id.
  """
    membership = _messages.MessageField('Membership', 1)
    membershipId = _messages.StringField(2)