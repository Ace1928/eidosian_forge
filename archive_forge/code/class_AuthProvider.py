from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthProvider(_messages.Message):
    """Configuration for an anthentication provider, including support for
  [JSON Web Token (JWT)](https://tools.ietf.org/html/draft-ietf-oauth-json-
  web-token-32).

  Fields:
    id: The unique identifier of the auth provider. It will be referred to by
      `AuthRequirement.provider_id`.  Example: "bookstore_auth".
    issuer: Identifies the principal that issued the JWT. See
      https://tools.ietf.org/html/draft-ietf-oauth-json-web-
      token-32#section-4.1.1 Usually a URL or an email address.  Example:
      https://securetoken.google.com Example:
      1234567-compute@developer.gserviceaccount.com
    jwksUri: URL of the provider's public key set to validate signature of the
      JWT. See [OpenID Discovery](https://openid.net/specs/openid-connect-
      discovery-1_0.html#ProviderMetadata). Optional if the key set document:
      - can be retrieved from    [OpenID Discovery](https://openid.net/specs
      /openid-connect-discovery-1_0.html    of the issuer.  - can be inferred
      from the email domain of the issuer (e.g. a Google service account).
      Example: https://www.googleapis.com/oauth2/v1/certs
  """
    id = _messages.StringField(1)
    issuer = _messages.StringField(2)
    jwksUri = _messages.StringField(3)