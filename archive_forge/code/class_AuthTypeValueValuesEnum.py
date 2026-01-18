from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthTypeValueValuesEnum(_messages.Enum):
    """Type of auth scheme.

    Values:
      AUTH_TYPE_UNSPECIFIED: <no description>
      NO_AUTH: No Auth.
      API_KEY_AUTH: API Key Auth.
      HTTP_BASIC_AUTH: HTTP Basic Auth.
      GOOGLE_SERVICE_ACCOUNT_AUTH: Google Service Account Auth.
      OAUTH: OAuth auth.
      OIDC_AUTH: OpenID Connect (OIDC) Auth.
    """
    AUTH_TYPE_UNSPECIFIED = 0
    NO_AUTH = 1
    API_KEY_AUTH = 2
    HTTP_BASIC_AUTH = 3
    GOOGLE_SERVICE_ACCOUNT_AUTH = 4
    OAUTH = 5
    OIDC_AUTH = 6