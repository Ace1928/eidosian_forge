from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesRefreshWorkspaceRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesRefreshWorkspaceRequest object.

  Fields:
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    projectId: The ID of the project.
    refreshWorkspaceRequest: A RefreshWorkspaceRequest resource to be passed
      as the request body.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    name = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    refreshWorkspaceRequest = _messages.MessageField('RefreshWorkspaceRequest', 3)
    repoName = _messages.StringField(4, required=True)