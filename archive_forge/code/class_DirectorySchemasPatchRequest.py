from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectorySchemasPatchRequest(_messages.Message):
    """A DirectorySchemasPatchRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    schema: A Schema resource to be passed as the request body.
    schemaKey: Name or immutable ID of the schema.
  """
    customerId = _messages.StringField(1, required=True)
    schema = _messages.MessageField('Schema', 2)
    schemaKey = _messages.StringField(3, required=True)