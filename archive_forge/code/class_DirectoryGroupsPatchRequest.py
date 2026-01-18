from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryGroupsPatchRequest(_messages.Message):
    """A DirectoryGroupsPatchRequest object.

  Fields:
    group: A Group resource to be passed as the request body.
    groupKey: Email or immutable ID of the group. If ID, it should match with
      id of group object
  """
    group = _messages.MessageField('Group', 1)
    groupKey = _messages.StringField(2, required=True)