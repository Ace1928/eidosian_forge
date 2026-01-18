from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUptimeCheckIpsResponse(_messages.Message):
    """The protocol for the ListUptimeCheckIps response.

  Fields:
    nextPageToken: This field represents the pagination token to retrieve the
      next page of results. If the value is empty, it means no further results
      for the request. To retrieve the next page of results, the value of the
      next_page_token is passed to the subsequent List method call (in the
      request message's page_token field). NOTE: this field is not yet
      implemented
    uptimeCheckIps: The returned list of IP addresses (including region and
      location) that the checkers run from.
  """
    nextPageToken = _messages.StringField(1)
    uptimeCheckIps = _messages.MessageField('UptimeCheckIp', 2, repeated=True)