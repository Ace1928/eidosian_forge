from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Notifications(_messages.Message):
    """A list of notification subscriptions.

  Fields:
    items: The list of items.
    kind: The kind of item this is. For lists of notifications, this is always
      storage#notifications.
  """
    items = _messages.MessageField('Notification', 1, repeated=True)
    kind = _messages.StringField(2, default=u'storage#notifications')