from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV2GitLabConfig(_messages.Message):
    """Configuration for connections to gitlab.com or an instance of GitLab
  Enterprise.

  Fields:
    authorizerCredential: Required. A GitLab personal access token with the
      `api` scope access.
    hostUri: The URI of the GitLab Enterprise host this connection is for. If
      not specified, the default value is https://gitlab.com.
    readAuthorizerCredential: Required. A GitLab personal access token with
      the minimum `read_api` scope access.
    serverVersion: Output only. Version of the GitLab Enterprise server
      running on the `host_uri`.
    serviceDirectoryConfig: Configuration for using Service Directory to
      privately connect to a GitLab Enterprise server. This should only be set
      if the GitLab Enterprise server is hosted on-premises and not reachable
      by public internet. If this field is left empty, calls to the GitLab
      Enterprise server will be made over the public internet.
    sslCa: SSL certificate to use for requests to GitLab Enterprise.
    webhookSecretSecretVersion: Required. Immutable. SecretManager resource
      containing the webhook secret of a GitLab Enterprise project, formatted
      as `projects/*/secrets/*/versions/*`.
  """
    authorizerCredential = _messages.MessageField('UserCredential', 1)
    hostUri = _messages.StringField(2)
    readAuthorizerCredential = _messages.MessageField('UserCredential', 3)
    serverVersion = _messages.StringField(4)
    serviceDirectoryConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV2ServiceDirectoryConfig', 5)
    sslCa = _messages.StringField(6)
    webhookSecretSecretVersion = _messages.StringField(7)