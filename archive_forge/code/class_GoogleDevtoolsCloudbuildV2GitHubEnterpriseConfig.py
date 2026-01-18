from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV2GitHubEnterpriseConfig(_messages.Message):
    """Configuration for connections to an instance of GitHub Enterprise.

  Fields:
    apiKey: Required. API Key used for authentication of webhook events.
    appId: Id of the GitHub App created from the manifest.
    appInstallationId: ID of the installation of the GitHub App.
    appSlug: The URL-friendly name of the GitHub App.
    authorizerCredential: OAuth credential of the account that authorized the
      Cloud Build GitHub App. It is recommended to use a robot account instead
      of a human user account The OAuth token must be tied to the Cloud Build
      GitHub App.
    hostUri: Required. The URI of the GitHub Enterprise host this connection
      is for.
    oauthClientIdSecretVersion: SecretManager resource containing the OAuth
      client_id of the GitHub App, formatted as
      `projects/*/secrets/*/versions/*`.
    oauthSecretSecretVersion: SecretManager resource containing the OAuth
      secret of the GitHub App, formatted as
      `projects/*/secrets/*/versions/*`.
    privateKeySecretVersion: SecretManager resource containing the private key
      of the GitHub App, formatted as `projects/*/secrets/*/versions/*`.
    serverVersion: Output only. GitHub Enterprise version installed at the
      host_uri.
    serviceDirectoryConfig: Configuration for using Service Directory to
      privately connect to a GitHub Enterprise server. This should only be set
      if the GitHub Enterprise server is hosted on-premises and not reachable
      by public internet. If this field is left empty, calls to the GitHub
      Enterprise server will be made over the public internet.
    sslCa: SSL certificate to use for requests to GitHub Enterprise.
    webhookSecretSecretVersion: SecretManager resource containing the webhook
      secret of the GitHub App, formatted as
      `projects/*/secrets/*/versions/*`.
  """
    apiKey = _messages.StringField(1)
    appId = _messages.IntegerField(2)
    appInstallationId = _messages.IntegerField(3)
    appSlug = _messages.StringField(4)
    authorizerCredential = _messages.MessageField('OAuthCredential', 5)
    hostUri = _messages.StringField(6)
    oauthClientIdSecretVersion = _messages.StringField(7)
    oauthSecretSecretVersion = _messages.StringField(8)
    privateKeySecretVersion = _messages.StringField(9)
    serverVersion = _messages.StringField(10)
    serviceDirectoryConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV2ServiceDirectoryConfig', 11)
    sslCa = _messages.StringField(12)
    webhookSecretSecretVersion = _messages.StringField(13)