from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchDirectGroupsResponse(_messages.Message):
    """The response message for MembershipsService.SearchDirectGroups.

  Fields:
    memberships: List of direct groups satisfying the query.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results available for listing.
  """
    memberships = _messages.MessageField('MembershipRelation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)