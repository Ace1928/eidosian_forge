from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPowerNetworksResponse(_messages.Message):
    """Response message containing the list of Power networks.

  Fields:
    nextPageToken: A token identifying a page of results from the server.
    powerNetworks: The list of networks.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    powerNetworks = _messages.MessageField('PowerNetwork', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)