from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwarePlatformConfig(_messages.Message):
    """VmwarePlatformConfig represents configuration for the VMware platform.

  Fields:
    bundles: Output only. The list of bundles installed in the admin cluster.
    platformVersion: Output only. The platform version e.g. 1.13.2.
    requiredPlatformVersion: Input only. The required platform version e.g.
      1.13.1. If the current platform version is lower than the target
      version, the platform version will be updated to the target version. If
      the target version is not installed in the platform (bundle versions),
      download the target version bundle.
    status: Output only. Resource status for the platform.
  """
    bundles = _messages.MessageField('VmwareBundleConfig', 1, repeated=True)
    platformVersion = _messages.StringField(2)
    requiredPlatformVersion = _messages.StringField(3)
    status = _messages.MessageField('ResourceStatus', 4)