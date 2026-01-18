from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserUndelete(_messages.Message):
    """JSON request template to undelete a user in Directory API.

  Fields:
    orgUnitPath: OrgUnit of User
  """
    orgUnitPath = _messages.StringField(1)