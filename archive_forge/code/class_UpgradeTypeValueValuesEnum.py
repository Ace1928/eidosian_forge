from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of upgrade being performed on the private cloud.

    Values:
      UPGRADE_TYPE_UNSPECIFIED: The default value. This value should never be
        used.
      VSPHERE_UPGRADE: Upgrade of vmware components when a major version is
        available. 7.0u2 -> 7.0u3.
      VSPHERE_PATCH: Patching of vmware components when a minor version is
        available. 7.0u2c -> 7.0u2d.
      VSPHERE_WORKAROUND: Workarounds to be applied on components for security
        fixes or otherwise.
      NON_VSPHERE_WORKAROUND: Workarounds to be applied for specific changes
        at PC level. eg: change in DRS rules, etc.
      ADHOC_JOB: Maps to on demand job. eg: scripts to be run against
        components
      FIRMWARE_UPGRADE: Placeholder for Firmware upgrades.
      SWITCH_UPGRADE: Placeholder for switch upgrades.
    """
    UPGRADE_TYPE_UNSPECIFIED = 0
    VSPHERE_UPGRADE = 1
    VSPHERE_PATCH = 2
    VSPHERE_WORKAROUND = 3
    NON_VSPHERE_WORKAROUND = 4
    ADHOC_JOB = 5
    FIRMWARE_UPGRADE = 6
    SWITCH_UPGRADE = 7