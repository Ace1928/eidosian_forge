from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosTestSetup(_messages.Message):
    """A description of how to set up an iOS device prior to running the test.

  Fields:
    additionalIpas: iOS apps to install in addition to those being directly
      tested.
    networkProfile: The network traffic profile used for running the test.
      Available network profiles can be queried by using the
      NETWORK_CONFIGURATION environment type when calling
      TestEnvironmentDiscoveryService.GetTestEnvironmentCatalog.
    pullDirectories: List of directories on the device to upload to Cloud
      Storage at the end of the test. Directories should either be in a shared
      directory (such as /private/var/mobile/Media) or within an accessible
      directory inside the app's filesystem (such as /Documents) by specifying
      the bundle ID.
    pushFiles: List of files to push to the device before starting the test.
  """
    additionalIpas = _messages.MessageField('FileReference', 1, repeated=True)
    networkProfile = _messages.StringField(2)
    pullDirectories = _messages.MessageField('IosDeviceFile', 3, repeated=True)
    pushFiles = _messages.MessageField('IosDeviceFile', 4, repeated=True)