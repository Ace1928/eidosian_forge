from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOauthClientsResponse(_messages.Message):
    """Response message for ListOauthClients.

  Fields:
    nextPageToken: Optional. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    oauthClients: A list of oauth clients.
  """
    nextPageToken = _messages.StringField(1)
    oauthClients = _messages.MessageField('OauthClient', 2, repeated=True)