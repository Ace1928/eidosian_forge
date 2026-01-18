from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WipeDeviceUserResponse(_messages.Message):
    """Response message for wiping the user's account from the device.

  Fields:
    deviceUser: Resultant DeviceUser object for the action.
  """
    deviceUser = _messages.MessageField('DeviceUser', 1)