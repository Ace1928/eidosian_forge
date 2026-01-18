from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMembershipBindingsResponse(_messages.Message):
    """List of MembershipBindings.

  Fields:
    membershipBindings: The list of membership_bindings
    nextPageToken: A token to request the next page of resources from the
      `ListMembershipBindings` method. The value of an empty string means that
      there are no more resources to return.
  """
    membershipBindings = _messages.MessageField('MembershipBinding', 1, repeated=True)
    nextPageToken = _messages.StringField(2)