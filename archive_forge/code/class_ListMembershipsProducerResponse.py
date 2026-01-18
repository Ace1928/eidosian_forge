from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMembershipsProducerResponse(_messages.Message):
    """Response schema for ListMembershipsProducer.

  Fields:
    memberships: List of the memberships in a given fleet.
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    totalSize: The number of memberships in the response.
  """
    memberships = _messages.MessageField('Membership', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3)