from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidDeviceList(_messages.Message):
    """A list of Android device configurations in which the test is to be
  executed.

  Fields:
    androidDevices: Required. A list of Android devices.
  """
    androidDevices = _messages.MessageField('AndroidDevice', 1, repeated=True)