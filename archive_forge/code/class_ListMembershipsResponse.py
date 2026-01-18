from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMembershipsResponse(_messages.Message):
    """The response message for MembershipsService.ListMemberships.

  Fields:
    memberships: The `Membership`s under the specified `parent`.
    nextPageToken: A continuation token to retrieve the next page of results,
      or empty if there are no more results available.
  """
    memberships = _messages.MessageField('Membership', 1, repeated=True)
    nextPageToken = _messages.StringField(2)