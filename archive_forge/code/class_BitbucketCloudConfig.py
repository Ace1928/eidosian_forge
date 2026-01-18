from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketCloudConfig(_messages.Message):
    """Configuration for connections to Bitbucket Cloud.

  Fields:
    authorizerCredential: Required. An access token with the `webhook`,
      `repository`, `repository:admin` and `pullrequest` scope access. It can
      be either a workspace, project or repository access token. It's
      recommended to use a system account to generate these credentials.
    readAuthorizerCredential: Required. An access token with the `repository`
      access. It can be either a workspace, project or repository access
      token. It's recommended to use a system account to generate the
      credentials.
    webhookSecretSecretVersion: Required. SecretManager resource containing
      the webhook secret used to verify webhook events, formatted as
      `projects/*/secrets/*/versions/*`.
    workspace: Required. The Bitbucket Cloud Workspace ID to be connected to
      Google Cloud Platform.
  """
    authorizerCredential = _messages.MessageField('UserCredential', 1)
    readAuthorizerCredential = _messages.MessageField('UserCredential', 2)
    webhookSecretSecretVersion = _messages.StringField(3)
    workspace = _messages.StringField(4)