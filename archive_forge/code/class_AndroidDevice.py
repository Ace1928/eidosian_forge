from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidDevice(_messages.Message):
    """A single Android device.

  Fields:
    androidModelId: Required. The id of the Android device to be used. Use the
      TestEnvironmentDiscoveryService to get supported options.
    androidVersionId: Required. The id of the Android OS version to be used.
      Use the TestEnvironmentDiscoveryService to get supported options.
    locale: Required. The locale the test device used for testing. Use the
      TestEnvironmentDiscoveryService to get supported options.
    orientation: Required. How the device is oriented during the test. Use the
      TestEnvironmentDiscoveryService to get supported options.
  """
    androidModelId = _messages.StringField(1)
    androidVersionId = _messages.StringField(2)
    locale = _messages.StringField(3)
    orientation = _messages.StringField(4)