from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OidcToken(_messages.Message):
    """Contains information needed for generating an [OpenID Connect
  token](https://developers.google.com/identity/protocols/OpenIDConnect). This
  type of authorization can be used for many scenarios, including calling
  Cloud Run, or endpoints where you intend to validate the token yourself.

  Fields:
    audience: Audience to be used when generating OIDC token. If not
      specified, the URI specified in target will be used.
    serviceAccountEmail: [Service account
      email](https://cloud.google.com/iam/docs/service-accounts) to be used
      for generating OIDC token. The service account must be within the same
      project as the queue. The caller must have iam.serviceAccounts.actAs
      permission for the service account.
  """
    audience = _messages.StringField(1)
    serviceAccountEmail = _messages.StringField(2)