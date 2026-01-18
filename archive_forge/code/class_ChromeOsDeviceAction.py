from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChromeOsDeviceAction(_messages.Message):
    """JSON request template for firing actions on ChromeOs Device in Directory

  Devices API.

  Fields:
    action: Action to be taken on the ChromeOs Device
    deprovisionReason: A string attribute.
  """
    action = _messages.StringField(1)
    deprovisionReason = _messages.StringField(2)