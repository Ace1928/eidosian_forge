from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesFilesGetRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesFilesGetRequest object.

  Enums:
    SourceContextCloudRepoAliasContextKindValueValuesEnum: The alias kind.
    SourceContextGerritAliasContextKindValueValuesEnum: The alias kind.

  Fields:
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    pageSize: The maximum number of values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page, or if using start_index.
    path: Path to the file or directory from the root directory of the source
      context. It must not have leading or trailing slashes.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    sourceContext_cloudRepo_aliasContext_kind: The alias kind.
    sourceContext_cloudRepo_aliasContext_name: The alias name.
    sourceContext_cloudRepo_aliasName: The name of an alias (branch, tag,
      etc.).
    sourceContext_cloudRepo_repoId_projectRepoId_projectId: The ID of the
      project.
    sourceContext_cloudRepo_repoId_projectRepoId_repoName: The name of the
      repo. Leave empty for the default repo.
    sourceContext_cloudRepo_repoId_uid: A server-assigned, globally unique
      identifier.
    sourceContext_cloudRepo_revisionId: A revision ID.
    sourceContext_cloudWorkspace_snapshotId: The ID of the snapshot. An empty
      snapshot_id refers to the most recent snapshot.
    sourceContext_cloudWorkspace_workspaceId_repoId_uid: A server-assigned,
      globally unique identifier.
    sourceContext_gerrit_aliasContext_kind: The alias kind.
    sourceContext_gerrit_aliasContext_name: The alias name.
    sourceContext_gerrit_aliasName: The name of an alias (branch, tag, etc.).
    sourceContext_gerrit_gerritProject: The full project name within the host.
      Projects may be nested, so "project/subproject" is a valid project name.
      The "repo name" is hostURI/project.
    sourceContext_gerrit_hostUri: The URI of a running Gerrit instance.
    sourceContext_gerrit_revisionId: A revision (commit) ID.
    sourceContext_git_revisionId: Git commit hash. required.
    sourceContext_git_url: Git repository URL.
    startPosition: If path refers to a file, the position of the first byte of
      its contents to return. If path refers to a directory, the position of
      the first entry in the listing. If page_token is specified, this field
      is ignored.
  """

    class SourceContextCloudRepoAliasContextKindValueValuesEnum(_messages.Enum):
        """The alias kind.

    Values:
      ANY: <no description>
      FIXED: <no description>
      MOVABLE: <no description>
      OTHER: <no description>
    """
        ANY = 0
        FIXED = 1
        MOVABLE = 2
        OTHER = 3

    class SourceContextGerritAliasContextKindValueValuesEnum(_messages.Enum):
        """The alias kind.

    Values:
      ANY: <no description>
      FIXED: <no description>
      MOVABLE: <no description>
      OTHER: <no description>
    """
        ANY = 0
        FIXED = 1
        MOVABLE = 2
        OTHER = 3
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2)
    pageToken = _messages.StringField(3)
    path = _messages.StringField(4, required=True)
    projectId = _messages.StringField(5, required=True)
    repoName = _messages.StringField(6, required=True)
    sourceContext_cloudRepo_aliasContext_kind = _messages.EnumField('SourceContextCloudRepoAliasContextKindValueValuesEnum', 7)
    sourceContext_cloudRepo_aliasContext_name = _messages.StringField(8)
    sourceContext_cloudRepo_aliasName = _messages.StringField(9)
    sourceContext_cloudRepo_repoId_projectRepoId_projectId = _messages.StringField(10)
    sourceContext_cloudRepo_repoId_projectRepoId_repoName = _messages.StringField(11)
    sourceContext_cloudRepo_repoId_uid = _messages.StringField(12)
    sourceContext_cloudRepo_revisionId = _messages.StringField(13)
    sourceContext_cloudWorkspace_snapshotId = _messages.StringField(14)
    sourceContext_cloudWorkspace_workspaceId_repoId_uid = _messages.StringField(15)
    sourceContext_gerrit_aliasContext_kind = _messages.EnumField('SourceContextGerritAliasContextKindValueValuesEnum', 16)
    sourceContext_gerrit_aliasContext_name = _messages.StringField(17)
    sourceContext_gerrit_aliasName = _messages.StringField(18)
    sourceContext_gerrit_gerritProject = _messages.StringField(19)
    sourceContext_gerrit_hostUri = _messages.StringField(20)
    sourceContext_gerrit_revisionId = _messages.StringField(21)
    sourceContext_git_revisionId = _messages.StringField(22)
    sourceContext_git_url = _messages.StringField(23)
    startPosition = _messages.IntegerField(24)