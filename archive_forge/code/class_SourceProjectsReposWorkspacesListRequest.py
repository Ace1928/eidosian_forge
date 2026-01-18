from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesListRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesListRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which parts of the Workspace resource
      should be returned in the response.

  Fields:
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
    view: Specifies which parts of the Workspace resource should be returned
      in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which parts of the Workspace resource should be returned in
    the response.

    Values:
      STANDARD: <no description>
      MINIMAL: <no description>
      FULL: <no description>
    """
        STANDARD = 0
        MINIMAL = 1
        FULL = 2
    projectId = _messages.StringField(1, required=True)
    repoId_uid = _messages.StringField(2)
    repoName = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)