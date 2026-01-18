from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationTypeValueValuesEnum(_messages.Enum):
    """Indicates the type of authorization.

    Values:
      AUTHORIZATION_TYPE_UNSPECIFIED: Type unspecified.
      AUTHORIZATION_CODE: Use OAuth 2 authorization codes that can be
        exchanged for a refresh token on the backend.
      GOOGLE_PLUS_AUTHORIZATION_CODE: Return an authorization code for a given
        Google+ page that can then be exchanged for a refresh token on the
        backend.
      FIRST_PARTY_OAUTH: Use First Party OAuth.
    """
    AUTHORIZATION_TYPE_UNSPECIFIED = 0
    AUTHORIZATION_CODE = 1
    GOOGLE_PLUS_AUTHORIZATION_CODE = 2
    FIRST_PARTY_OAUTH = 3