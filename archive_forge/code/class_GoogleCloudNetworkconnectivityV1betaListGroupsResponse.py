from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaListGroupsResponse(_messages.Message):
    """Response for HubService.ListGroups method.

  Fields:
    groups: The requested groups.
    nextPageToken: The token for the next page of the response. To see more
      results, use this value as the page_token for your next request. If this
      value is empty, there are no more results.
    unreachable: Hubs that could not be reached.
  """
    groups = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)