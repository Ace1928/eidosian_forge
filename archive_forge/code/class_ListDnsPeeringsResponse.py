from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDnsPeeringsResponse(_messages.Message):
    """Response message for list DNS peerings.

  Fields:
    dnsPeerings: List of dns peering.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    dnsPeerings = _messages.MessageField('DnsPeering', 1, repeated=True)
    nextPageToken = _messages.StringField(2)