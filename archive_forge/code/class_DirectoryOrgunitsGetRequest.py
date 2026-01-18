from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryOrgunitsGetRequest(_messages.Message):
    """A DirectoryOrgunitsGetRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    orgUnitPath: Full path of the organizational unit or its ID
  """
    customerId = _messages.StringField(1, required=True)
    orgUnitPath = _messages.StringField(2, required=True)