from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerateAccessTokenResponse(_messages.Message):
    """Response message for NotebookInternalService.GenerateToken.

  Fields:
    accessToken: Short-lived access token string which may be used to access
      Google APIs.
    expiresIn: The time in seconds when the access token expires. Typically
      that's 3600.
    scope: Space-separated list of scopes contained in the returned token.
      https://cloud.google.com/docs/authentication/token-types#access-contents
    tokenType: Type of the returned access token (e.g. "Bearer"). It specifies
      how the token must be used. Bearer tokens may be used by any entity
      without proof of identity.
  """
    accessToken = _messages.StringField(1)
    expiresIn = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    scope = _messages.StringField(3)
    tokenType = _messages.StringField(4)