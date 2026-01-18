from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryNotificationsPatchRequest(_messages.Message):
    """A DirectoryNotificationsPatchRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account.
    notification: A Notification resource to be passed as the request body.
    notificationId: The unique ID of the notification.
  """
    customer = _messages.StringField(1, required=True)
    notification = _messages.MessageField('Notification', 2)
    notificationId = _messages.StringField(3, required=True)