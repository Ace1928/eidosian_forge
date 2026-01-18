from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabSecrets(_messages.Message):
    """GitLabSecrets represents the secrets in Secret Manager for a GitLab
  integration.

  Fields:
    apiAccessTokenVersion: Required. The resource name for the api access
      token's secret version
    apiKeyVersion: Required. Immutable. API Key that will be attached to
      webhook requests from GitLab to Cloud Build.
    readAccessTokenVersion: Required. The resource name for the read access
      token's secret version
    webhookSecretVersion: Required. Immutable. The resource name for the
      webhook secret's secret version. Once this field has been set, it cannot
      be changed. If you need to change it, please create another
      GitLabConfig.
  """
    apiAccessTokenVersion = _messages.StringField(1)
    apiKeyVersion = _messages.StringField(2)
    readAccessTokenVersion = _messages.StringField(3)
    webhookSecretVersion = _messages.StringField(4)