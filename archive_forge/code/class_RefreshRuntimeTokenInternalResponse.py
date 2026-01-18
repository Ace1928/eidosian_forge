from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RefreshRuntimeTokenInternalResponse(_messages.Message):
    """Response with a new access token.

  Fields:
    accessToken: The OAuth 2.0 access token.
    expireTime: Output only. Token expiration time.
  """
    accessToken = _messages.StringField(1)
    expireTime = _messages.StringField(2)