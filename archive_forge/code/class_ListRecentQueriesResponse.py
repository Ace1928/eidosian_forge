from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRecentQueriesResponse(_messages.Message):
    """The response from ListRecentQueries.

  Fields:
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
    recentQueries: A list of recent queries.
    unreachable: The unreachable resources. Each resource can be either 1) a
      saved query if a specific query is unreachable or 2) a location if a
      specific location is unreachable.
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/recentQueries/[QUERY_ID]"
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]" For
      example:"projects/my-project/locations/global/recentQueries/12345678"
      "projects/my-project/locations/global"If there are unreachable
      resources, the response will first return pages that contain recent
      queries, and then return pages that contain the unreachable resources.
  """
    nextPageToken = _messages.StringField(1)
    recentQueries = _messages.MessageField('RecentQuery', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)