from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModifyCloudToDeviceConfigRequest(_messages.Message):
    """Request for `ModifyCloudToDeviceConfig`.

  Fields:
    binaryData: Required. The configuration data for the device.
    versionToUpdate: The version number to update. If this value is zero, it
      will not check the version number of the server and will always update
      the current version; otherwise, this update will fail if the version
      number found on the server does not match this version number. This is
      used to support multiple simultaneous updates without losing data.
  """
    binaryData = _messages.BytesField(1)
    versionToUpdate = _messages.IntegerField(2)