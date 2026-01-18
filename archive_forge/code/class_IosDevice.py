from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosDevice(_messages.Message):
    """A single iOS device.

  Fields:
    iosModelId: Required. The id of the iOS device to be used. Use the
      TestEnvironmentDiscoveryService to get supported options.
    iosVersionId: Required. The id of the iOS major software version to be
      used. Use the TestEnvironmentDiscoveryService to get supported options.
    locale: Required. The locale the test device used for testing. Use the
      TestEnvironmentDiscoveryService to get supported options.
    orientation: Required. How the device is oriented during the test. Use the
      TestEnvironmentDiscoveryService to get supported options.
  """
    iosModelId = _messages.StringField(1)
    iosVersionId = _messages.StringField(2)
    locale = _messages.StringField(3)
    orientation = _messages.StringField(4)