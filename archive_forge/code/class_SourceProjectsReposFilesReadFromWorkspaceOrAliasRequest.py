from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposFilesReadFromWorkspaceOrAliasRequest(_messages.Message):
    """A SourceProjectsReposFilesReadFromWorkspaceOrAliasRequest object.

  Fields:
    alias: MOVABLE alias to read from, if the workspace doesn't exist.
    pageSize: The maximum number of values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page.
    path: Path to the file or directory from the root directory of the source
      context. It must not have leading or trailing slashes.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
    startPosition: If path refers to a file, the position of the first byte of
      its contents to return. If path refers to a directory, the position of
      the first entry in the listing. If page_token is specified, this field
      is ignored.
    workspaceName: Workspace to read from, if it exists.
  """
    alias = _messages.StringField(1)
    pageSize = _messages.IntegerField(2)
    pageToken = _messages.StringField(3)
    path = _messages.StringField(4, required=True)
    projectId = _messages.StringField(5, required=True)
    repoId_uid = _messages.StringField(6)
    repoName = _messages.StringField(7, required=True)
    startPosition = _messages.IntegerField(8)
    workspaceName = _messages.StringField(9)