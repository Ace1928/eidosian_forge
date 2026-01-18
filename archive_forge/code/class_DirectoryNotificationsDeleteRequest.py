from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryNotificationsDeleteRequest(_messages.Message):
    """A DirectoryNotificationsDeleteRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. The customerId
      is also returned as part of the Users resource.
    notificationId: The unique ID of the notification.
  """
    customer = _messages.StringField(1, required=True)
    notificationId = _messages.StringField(2, required=True)