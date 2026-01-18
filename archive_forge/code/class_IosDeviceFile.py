from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosDeviceFile(_messages.Message):
    """A file or directory to install on the device before the test starts.

  Fields:
    bundleId: The bundle id of the app where this file lives. iOS apps sandbox
      their own filesystem, so app files must specify which app installed on
      the device.
    content: The source file
    devicePath: Location of the file on the device, inside the app's sandboxed
      filesystem
  """
    bundleId = _messages.StringField(1)
    content = _messages.MessageField('FileReference', 2)
    devicePath = _messages.StringField(3)