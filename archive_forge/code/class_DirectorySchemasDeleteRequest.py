from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectorySchemasDeleteRequest(_messages.Message):
    """A DirectorySchemasDeleteRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    schemaKey: Name or immutable ID of the schema
  """
    customerId = _messages.StringField(1, required=True)
    schemaKey = _messages.StringField(2, required=True)