from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketDataCenterConfig(_messages.Message):
    """Configuration for connections to Bitbucket Data Center.

  Fields:
    authorizerCredential: Required. A http access token with the `REPO_ADMIN`
      scope access.
    hostUri: Required. The URI of the Bitbucket Data Center instance or
      cluster this connection is for.
    readAuthorizerCredential: Required. A http access token with the
      `REPO_READ` access.
    serverVersion: Output only. Version of the Bitbucket Data Center running
      on the `host_uri`.
    serviceDirectoryConfig: Optional. Configuration for using Service
      Directory to privately connect to a Bitbucket Data Center. This should
      only be set if the Bitbucket Data Center is hosted on-premises and not
      reachable by public internet. If this field is left empty, calls to the
      Bitbucket Data Center will be made over the public internet.
    sslCa: Optional. SSL certificate to use for requests to the Bitbucket Data
      Center.
    webhookSecretSecretVersion: Required. Immutable. SecretManager resource
      containing the webhook secret used to verify webhook events, formatted
      as `projects/*/secrets/*/versions/*`.
  """
    authorizerCredential = _messages.MessageField('UserCredential', 1)
    hostUri = _messages.StringField(2)
    readAuthorizerCredential = _messages.MessageField('UserCredential', 3)
    serverVersion = _messages.StringField(4)
    serviceDirectoryConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV2ServiceDirectoryConfig', 5)
    sslCa = _messages.StringField(6)
    webhookSecretSecretVersion = _messages.StringField(7)