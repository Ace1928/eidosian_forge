from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WipeDeviceResponse(_messages.Message):
    """Response message for wiping all data on the device.

  Fields:
    device: Resultant Device object for the action. Note that asset tags will
      not be returned in the device object.
  """
    device = _messages.MessageField('Device', 1)