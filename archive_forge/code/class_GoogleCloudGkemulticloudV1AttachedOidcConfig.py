from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedOidcConfig(_messages.Message):
    """OIDC discovery information of the target cluster. Kubernetes Service
  Account (KSA) tokens are JWT tokens signed by the cluster API server. This
  fields indicates how Google Cloud Platform services validate KSA tokens in
  order to allow system workloads (such as GKE Connect and telemetry agents)
  to authenticate back to Google Cloud Platform. Both clusters with public and
  private issuer URLs are supported. Clusters with public issuers only need to
  specify the `issuer_url` field while clusters with private issuers need to
  provide both `issuer_url` and `oidc_jwks`.

  Fields:
    issuerUrl: A JSON Web Token (JWT) issuer URI. `issuer` must start with
      `https://`.
    jwks: Optional. OIDC verification keys in JWKS format (RFC 7517). It
      contains a list of OIDC verification keys that can be used to verify
      OIDC JWTs. This field is required for cluster that doesn't have a
      publicly available discovery endpoint. When provided, it will be
      directly used to verify the OIDC JWT asserted by the IDP.
  """
    issuerUrl = _messages.StringField(1)
    jwks = _messages.BytesField(2)