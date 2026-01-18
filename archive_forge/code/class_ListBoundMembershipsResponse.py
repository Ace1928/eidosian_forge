from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBoundMembershipsResponse(_messages.Message):
    """List of Memberships bound to a Scope.

  Fields:
    memberships: The list of Memberships bound to the given Scope.
    nextPageToken: A token to request the next page of resources from the
      `ListBoundMemberships` method. The value of an empty string means that
      there are no more resources to return.
    unreachable: List of locations that could not be reached while fetching
      this list.
  """
    memberships = _messages.MessageField('Membership', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)