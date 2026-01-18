from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeployedIndexAuthConfigAuthProvider(_messages.Message):
    """Configuration for an authentication provider, including support for
  [JSON Web Token (JWT)](https://tools.ietf.org/html/draft-ietf-oauth-json-
  web-token-32).

  Fields:
    allowedIssuers: A list of allowed JWT issuers. Each entry must be a valid
      Google service account, in the following format: `service-account-
      name@project-id.iam.gserviceaccount.com`
    audiences: The list of JWT [audiences](https://tools.ietf.org/html/draft-
      ietf-oauth-json-web-token-32#section-4.1.3). that are allowed to access.
      A JWT containing any of these audiences will be accepted.
  """
    allowedIssuers = _messages.StringField(1, repeated=True)
    audiences = _messages.StringField(2, repeated=True)