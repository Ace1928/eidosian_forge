from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGroupsResponse(_messages.Message):
    """Response message for ListGroups operation.

  Fields:
    groups: Groups returned in response to list request. The results are not
      sorted.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results available for listing.
  """
    groups = _messages.MessageField('Group', 1, repeated=True)
    nextPageToken = _messages.StringField(2)