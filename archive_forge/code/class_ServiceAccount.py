from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAccount(_messages.Message):
    """A subscription to receive Google PubSub notifications.

  Fields:
    email_address: The ID of the notification.
    kind: The kind of item this is. For notifications, this is always
      storage#notification.
  """
    email_address = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'storage#serviceAccount')