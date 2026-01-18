from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringUptimeCheckIpsListRequest(_messages.Message):
    """A MonitoringUptimeCheckIpsListRequest object.

  Fields:
    pageSize: The maximum number of results to return in a single response.
      The server may further constrain the maximum number of results returned
      in a single page. If the page_size is <=0, the server will decide the
      number of results to be returned. NOTE: this field is not yet
      implemented
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return more results from the previous
      method call. NOTE: this field is not yet implemented
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)