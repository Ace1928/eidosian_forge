from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListIdentityAwareProxyClientsResponse(_messages.Message):
    """Response message for ListIdentityAwareProxyClients.

  Fields:
    identityAwareProxyClients: Clients existing in the brand.
    nextPageToken: A token, which can be send as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    identityAwareProxyClients = _messages.MessageField('IdentityAwareProxyClient', 1, repeated=True)
    nextPageToken = _messages.StringField(2)