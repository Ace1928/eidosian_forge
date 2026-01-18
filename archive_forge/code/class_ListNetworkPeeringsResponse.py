from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNetworkPeeringsResponse(_messages.Message):
    """Response message for VmwareEngine.ListNetworkPeerings

  Fields:
    networkPeerings: A list of network peerings.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Unreachable resources.
  """
    networkPeerings = _messages.MessageField('NetworkPeering', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)