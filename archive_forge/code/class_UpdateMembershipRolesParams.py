from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateMembershipRolesParams(_messages.Message):
    """The details of an update to a `MembershipRole`.

  Fields:
    fieldMask: The fully-qualified names of fields to update. May only contain
      the field `expiry_detail.expire_time`.
    membershipRole: The `MembershipRole`s to be updated. Only `MEMBER`
      `MembershipRole` can currently be updated.
  """
    fieldMask = _messages.StringField(1)
    membershipRole = _messages.MessageField('MembershipRole', 2)