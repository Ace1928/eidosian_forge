from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposAliasesDeleteRequest(_messages.Message):
    """A SourceProjectsReposAliasesDeleteRequest object.

  Enums:
    KindValueValuesEnum: The kind of the alias to delete.

  Fields:
    kind: The kind of the alias to delete.
    name: The name of the alias to delete.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
    revisionId: If non-empty, must match the revision that the alias refers
      to.
  """

    class KindValueValuesEnum(_messages.Enum):
        """The kind of the alias to delete.

    Values:
      ANY: <no description>
      FIXED: <no description>
      MOVABLE: <no description>
      MERCURIAL_BRANCH_DEPRECATED: <no description>
      OTHER: <no description>
      SPECIAL_DEPRECATED: <no description>
    """
        ANY = 0
        FIXED = 1
        MOVABLE = 2
        MERCURIAL_BRANCH_DEPRECATED = 3
        OTHER = 4
        SPECIAL_DEPRECATED = 5
    kind = _messages.EnumField('KindValueValuesEnum', 1, required=True)
    name = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    repoId_uid = _messages.StringField(4)
    repoName = _messages.StringField(5, required=True)
    revisionId = _messages.StringField(6)