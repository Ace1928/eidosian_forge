from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMulticastConsumerAssociationsResponse(_messages.Message):
    """Response message for ListMulticastConsumerAssociations.

  Fields:
    multicastConsumerAssociations: The list of multicast consumer
      associations.
    nextPageToken: A page token from an earlier query, as returned in
      `next_page_token`.
    unreachable: Locations that could not be reached.
  """
    multicastConsumerAssociations = _messages.MessageField('MulticastConsumerAssociation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)