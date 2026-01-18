from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidMatrix(_messages.Message):
    """A set of Android device configuration permutations is defined by the the
  cross-product of the given axes. Internally, the given AndroidMatrix will be
  expanded into a set of AndroidDevices. Only supported permutations will be
  instantiated. Invalid permutations (e.g., incompatible models/versions) are
  ignored.

  Fields:
    androidModelIds: Required. The ids of the set of Android device to be
      used. Use the TestEnvironmentDiscoveryService to get supported options.
    androidVersionIds: Required. The ids of the set of Android OS version to
      be used. Use the TestEnvironmentDiscoveryService to get supported
      options.
    locales: Required. The set of locales the test device will enable for
      testing. Use the TestEnvironmentDiscoveryService to get supported
      options.
    orientations: Required. The set of orientations to test with. Use the
      TestEnvironmentDiscoveryService to get supported options.
  """
    androidModelIds = _messages.StringField(1, repeated=True)
    androidVersionIds = _messages.StringField(2, repeated=True)
    locales = _messages.StringField(3, repeated=True)
    orientations = _messages.StringField(4, repeated=True)