from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsGitLabConfigsRemoveGitLabConnectedRepositoryRequest(_messages.Message):
    """A CloudbuildProjectsLocationsGitLabConfigsRemoveGitLabConnectedRepositor
  yRequest object.

  Fields:
    config: Required. The name of the `GitLabConfig` to remove a connected
      repository. Format:
      `projects/{project}/locations/{location}/gitLabConfigs/{config}`
    removeGitLabConnectedRepositoryRequest: A
      RemoveGitLabConnectedRepositoryRequest resource to be passed as the
      request body.
  """
    config = _messages.StringField(1, required=True)
    removeGitLabConnectedRepositoryRequest = _messages.MessageField('RemoveGitLabConnectedRepositoryRequest', 2)