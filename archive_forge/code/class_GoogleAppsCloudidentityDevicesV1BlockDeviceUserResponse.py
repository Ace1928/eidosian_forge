from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1BlockDeviceUserResponse(_messages.Message):
    """Response message for blocking the device from accessing user data.

  Fields:
    deviceUser: Resultant DeviceUser object for the action.
  """
    deviceUser = _messages.MessageField('GoogleAppsCloudidentityDevicesV1DeviceUser', 1)