from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1CancelWipeDeviceUserResponse(_messages.Message):
    """Response message for cancelling an unfinished user account wipe.

  Fields:
    deviceUser: Resultant DeviceUser object for the action.
  """
    deviceUser = _messages.MessageField('GoogleAppsCloudidentityDevicesV1DeviceUser', 1)