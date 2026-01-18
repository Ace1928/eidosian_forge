from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaListRouteTablesResponse(_messages.Message):
    """Response for HubService.ListRouteTables method.

  Fields:
    nextPageToken: The token for the next page of the response. To see more
      results, use this value as the page_token for your next request. If this
      value is empty, there are no more results.
    routeTables: The requested route tables.
    unreachable: Hubs that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    routeTables = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaRouteTable', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)