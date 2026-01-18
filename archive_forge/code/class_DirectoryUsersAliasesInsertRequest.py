from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersAliasesInsertRequest(_messages.Message):
    """A DirectoryUsersAliasesInsertRequest object.

  Fields:
    alias: A Alias resource to be passed as the request body.
    userKey: Email or immutable ID of the user
  """
    alias = _messages.MessageField('Alias', 1)
    userKey = _messages.StringField(2, required=True)