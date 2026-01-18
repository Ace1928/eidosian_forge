from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsGitLabConfigsConnectedRepositoriesBatchCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsGitLabConfigsConnectedRepositoriesBatchCrea
  teRequest object.

  Fields:
    batchCreateGitLabConnectedRepositoriesRequest: A
      BatchCreateGitLabConnectedRepositoriesRequest resource to be passed as
      the request body.
    parent: The name of the `GitLabConfig` that adds connected repositories.
      Format: `projects/{project}/locations/{location}/gitLabConfigs/{config}`
  """
    batchCreateGitLabConnectedRepositoriesRequest = _messages.MessageField('BatchCreateGitLabConnectedRepositoriesRequest', 1)
    parent = _messages.StringField(2, required=True)