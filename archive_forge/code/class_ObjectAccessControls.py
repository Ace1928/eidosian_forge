from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectAccessControls(_messages.Message):
    """An access-control list.

  Fields:
    items: The list of items.
    kind: The kind of item this is. For lists of object access control
      entries, this is always storage#objectAccessControls.
  """
    items = _messages.MessageField('ObjectAccessControl', 1, repeated=True)
    kind = _messages.StringField(2, default=u'storage#objectAccessControls')