from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPeeringsResponse(_messages.Message):
    """ListPeeringsResponse is the response message for ListPeerings method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    peerings: A list of Managed Identities Service Peerings in the project.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    peerings = _messages.MessageField('Peering', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)