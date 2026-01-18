from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchStaticIpsResponse(_messages.Message):
    """Response message for a 'FetchStaticIps' request.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    staticIps: List of static IPs.
  """
    nextPageToken = _messages.StringField(1)
    staticIps = _messages.StringField(2, repeated=True)