from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesSnapshotsListRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesSnapshotsListRequest object.

  Fields:
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    pageSize: The maximum number of values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    workspaceId_repoId_uid: A server-assigned, globally unique identifier.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    repoName = _messages.StringField(5, required=True)
    workspaceId_repoId_uid = _messages.StringField(6)