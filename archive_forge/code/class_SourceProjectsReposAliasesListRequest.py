from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposAliasesListRequest(_messages.Message):
    """A SourceProjectsReposAliasesListRequest object.

  Enums:
    KindValueValuesEnum: Return only aliases of this kind.

  Fields:
    kind: Return only aliases of this kind.
    pageSize: The maximum number of values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Return only aliases of this kind.

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
    kind = _messages.EnumField('KindValueValuesEnum', 1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    repoId_uid = _messages.StringField(5)
    repoName = _messages.StringField(6, required=True)