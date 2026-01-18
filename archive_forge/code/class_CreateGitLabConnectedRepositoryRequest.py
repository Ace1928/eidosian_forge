from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateGitLabConnectedRepositoryRequest(_messages.Message):
    """Request to connect a repository from a connected GitLab host.

  Fields:
    gitlabConnectedRepository: Required. The GitLab repository to connect.
    parent: Required. The name of the `GitLabConfig` that adds connected
      repository. Format:
      `projects/{project}/locations/{location}/gitLabConfigs/{config}`
  """
    gitlabConnectedRepository = _messages.MessageField('GitLabConnectedRepository', 1)
    parent = _messages.StringField(2)