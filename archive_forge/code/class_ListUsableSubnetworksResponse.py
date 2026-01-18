from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUsableSubnetworksResponse(_messages.Message):
    """ListUsableSubnetworksResponse is the response of
  ListUsableSubnetworksRequest.

  Fields:
    nextPageToken: This token allows you to get the next page of results for
      list requests. If the number of results is larger than `page_size`, use
      the `next_page_token` as a value for the query parameter `page_token` in
      the next request. The value will become empty when there are no more
      pages.
    subnetworks: A list of usable subnetworks in the specified network
      project.
  """
    nextPageToken = _messages.StringField(1)
    subnetworks = _messages.MessageField('UsableSubnetwork', 2, repeated=True)