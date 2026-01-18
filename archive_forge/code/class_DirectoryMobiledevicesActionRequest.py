from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMobiledevicesActionRequest(_messages.Message):
    """A DirectoryMobiledevicesActionRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    mobileDeviceAction: A MobileDeviceAction resource to be passed as the
      request body.
    resourceId: Immutable ID of Mobile Device
  """
    customerId = _messages.StringField(1, required=True)
    mobileDeviceAction = _messages.MessageField('MobileDeviceAction', 2)
    resourceId = _messages.StringField(3, required=True)