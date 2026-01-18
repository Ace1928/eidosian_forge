from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketServerSecrets(_messages.Message):
    """BitbucketServerSecrets represents the secrets in Secret Manager for a
  Bitbucket Server.

  Fields:
    adminAccessTokenVersionName: Required. The resource name for the admin
      access token's secret version.
    readAccessTokenVersionName: Required. The resource name for the read
      access token's secret version.
    webhookSecretVersionName: Required. Immutable. The resource name for the
      webhook secret's secret version. Once this field has been set, it cannot
      be changed. If you need to change it, please create another
      BitbucketServerConfig.
  """
    adminAccessTokenVersionName = _messages.StringField(1)
    readAccessTokenVersionName = _messages.StringField(2)
    webhookSecretVersionName = _messages.StringField(3)