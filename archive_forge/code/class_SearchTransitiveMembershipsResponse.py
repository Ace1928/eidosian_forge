from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchTransitiveMembershipsResponse(_messages.Message):
    """The response message for MembershipsService.SearchTransitiveMemberships.

  Fields:
    memberships: List of transitive members satisfying the query.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
  """
    memberships = _messages.MessageField('MemberRelation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)