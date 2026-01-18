from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsOpenIdConfig(_messages.Message):
    """AwsOpenIdConfig is an OIDC discovery document for the cluster. See the
  OpenID Connect Discovery 1.0 specification for details.

  Fields:
    claims_supported: Supported claims.
    grant_types: Supported grant types.
    id_token_signing_alg_values_supported: supported ID Token signing
      Algorithms.
    issuer: OIDC Issuer.
    jwks_uri: JSON Web Key uri.
    response_types_supported: Supported response types.
    subject_types_supported: Supported subject types.
  """
    claims_supported = _messages.StringField(1, repeated=True)
    grant_types = _messages.StringField(2, repeated=True)
    id_token_signing_alg_values_supported = _messages.StringField(3, repeated=True)
    issuer = _messages.StringField(4)
    jwks_uri = _messages.StringField(5)
    response_types_supported = _messages.StringField(6, repeated=True)
    subject_types_supported = _messages.StringField(7, repeated=True)