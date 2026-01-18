from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChromeOsMoveDevicesToOu(_messages.Message):
    """JSON request template for moving ChromeOs Device to given OU in

  Directory Devices API.

  Fields:
    deviceIds: ChromeOs Devices to be moved to OU
  """
    deviceIds = _messages.StringField(1, repeated=True)