from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosDeviceList(_messages.Message):
    """A list of iOS device configurations in which the test is to be executed.

  Fields:
    iosDevices: Required. A list of iOS devices.
  """
    iosDevices = _messages.MessageField('IosDevice', 1, repeated=True)