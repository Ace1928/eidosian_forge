from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOauthClientCredentialsResponse(_messages.Message):
    """Response message for ListOauthClientCredentials.

  Fields:
    oauthClientCredentials: A list of oauth client credentials.
  """
    oauthClientCredentials = _messages.MessageField('OauthClientCredential', 1, repeated=True)