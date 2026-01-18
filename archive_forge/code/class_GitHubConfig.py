from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubConfig(_messages.Message):
    """Configuration for connections to github.com.

  Fields:
    appInstallationId: GitHub App installation id.
    authorizerCredential: OAuth credential of the account that authorized the
      Cloud Build GitHub App. It is recommended to use a robot account instead
      of a human user account. The OAuth token must be tied to the Cloud Build
      GitHub App.
  """
    appInstallationId = _messages.IntegerField(1)
    authorizerCredential = _messages.MessageField('OAuthCredential', 2)