from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAdminClusterMembershipsResponse(_messages.Message):
    """Response message for the `GkeHub.ListAdminClusterMemberships` method.

  Fields:
    adminClusterMemberships: The list of matching Memberships of admin
      clusters.
    nextPageToken: A token to request the next page of resources from the
      `ListAdminClusterMemberships` method. The value of an empty string means
      that there are no more resources to return.
    unreachable: List of locations that could not be reached while fetching
      this list.
  """
    adminClusterMemberships = _messages.MessageField('Membership', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)