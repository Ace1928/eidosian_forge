from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesModifyWorkspaceRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesModifyWorkspaceRequest object.

  Fields:
    modifyWorkspaceRequest: A ModifyWorkspaceRequest resource to be passed as
      the request body.
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    modifyWorkspaceRequest = _messages.MessageField('ModifyWorkspaceRequest', 1)
    name = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    repoName = _messages.StringField(4, required=True)