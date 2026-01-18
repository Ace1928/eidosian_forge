from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryGroupsAliasesInsertRequest(_messages.Message):
    """A DirectoryGroupsAliasesInsertRequest object.

  Fields:
    alias: A Alias resource to be passed as the request body.
    groupKey: Email or immutable ID of the group
  """
    alias = _messages.MessageField('Alias', 1)
    groupKey = _messages.StringField(2, required=True)