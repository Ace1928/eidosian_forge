from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectTeamValue(_messages.Message):
    """The project team associated with the entity, if any.

    Fields:
      projectNumber: The project number.
      team: The team.
    """
    projectNumber = _messages.StringField(1)
    team = _messages.StringField(2)