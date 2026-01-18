from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DirectAccessVersionInfo(_messages.Message):
    """Denotes whether Direct Access is supported, and by which client
  versions. DirectAccessService is currently available as a preview to select
  developers. You can register today on behalf of you and your team at
  https://developer.android.com/studio/preview/android-device-streaming

  Fields:
    directAccessSupported: Whether direct access is supported at all. Clients
      are expected to filter down the device list to only android models and
      versions which support Direct Access when that is the user intent.
    minimumAndroidStudioVersion: Output only. Indicates client-device
      compatibility, where a device is known to work only with certain
      workarounds implemented in the Android Studio client. Expected format
      "major.minor.micro.patch", e.g. "5921.22.2211.8881706".
  """
    directAccessSupported = _messages.BooleanField(1)
    minimumAndroidStudioVersion = _messages.StringField(2)