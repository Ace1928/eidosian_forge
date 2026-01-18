from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubInstallationsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsGithubInstallationsDeleteRequest object.

  Fields:
    installationId: GitHub app installation ID.
    name: The name of the `GitHubInstallation` to delete. Format:
      `projects/{project}/locations/{location}/installations/{installation}`
    projectId: Cloud Project ID.
  """
    installationId = _messages.IntegerField(1, required=True)
    name = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)