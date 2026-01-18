from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposRevisionsListRequest(_messages.Message):
    """A SourceProjectsReposRevisionsListRequest object.

  Enums:
    WalkDirectionValueValuesEnum: The direction to walk the graph.

  Fields:
    ends: Revision IDs (hexadecimal strings) that specify where the listing
      ends. If this field is present, the listing will contain only revisions
      that are topologically between starts and ends, inclusive.
    pageSize: The maximum number of values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page.
    path: List only those revisions that modify path.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
    starts: Revision IDs (hexadecimal strings) that specify where the listing
      begins. If empty, the repo heads (revisions with no children) are used.
    walkDirection: The direction to walk the graph.
  """

    class WalkDirectionValueValuesEnum(_messages.Enum):
        """The direction to walk the graph.

    Values:
      BACKWARD: <no description>
      FORWARD: <no description>
    """
        BACKWARD = 0
        FORWARD = 1
    ends = _messages.StringField(1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    path = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)
    repoId_uid = _messages.StringField(6)
    repoName = _messages.StringField(7, required=True)
    starts = _messages.StringField(8, repeated=True)
    walkDirection = _messages.EnumField('WalkDirectionValueValuesEnum', 9)