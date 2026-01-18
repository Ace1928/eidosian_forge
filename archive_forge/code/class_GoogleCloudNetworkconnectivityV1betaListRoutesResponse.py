from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaListRoutesResponse(_messages.Message):
    """Response for HubService.ListRoutes method.

  Fields:
    nextPageToken: The token for the next page of the response. To see more
      results, use this value as the page_token for your next request. If this
      value is empty, there are no more results.
    routes: The requested routes.
    unreachable: RouteTables that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    routes = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaRoute', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)