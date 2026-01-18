from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesDeleteRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesDeleteRequest object.

  Fields:
    currentSnapshotId: If non-empty, current_snapshot_id must refer to the
      most recent update to the workspace, or ABORTED is returned.
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    workspaceId_repoId_uid: A server-assigned, globally unique identifier.
  """
    currentSnapshotId = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    repoName = _messages.StringField(4, required=True)
    workspaceId_repoId_uid = _messages.StringField(5)