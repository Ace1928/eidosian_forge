from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeJob(_messages.Message):
    """Private cloud Upgrade Job resource.

  Enums:
    StateValueValuesEnum: Output only. The state of the resource.
    UpgradeTypeValueValuesEnum: Output only. The type of upgrade being
      performed on the private cloud.

  Fields:
    componentUpgrades: Output only. List of components that are being upgraded
      and their current status.
    createTime: Output only. Creation time of this resource. It also serves as
      start time of upgrade Job.
    endTime: Output only. The ending time of the upgrade Job. Only set when
      upgrade reaches a succeeded/failed/cancelled state.
    name: Output only. The resource name of the private cloud upgrade.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-west1-a/privateClouds/my-
      cloud/upgradeJobs/my-upgrade-job`
    progressPercent: Output only. Overall progress of the upgrade job in
      percentage (between 0-100%).
    startVersion: Output only. The starting version of the private cloud for
      this upgrade Job.
    state: Output only. The state of the resource.
    targetVersion: Output only. The targeted version of the private cloud at
      the end of upgrade Job.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    upgradeType: Output only. The type of upgrade being performed on the
      private cloud.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      RUNNING: Upgrade Job is in progress.
      PAUSED: Upgrade Job is paused. This happens between upgrade windows, or
        if pause upgrade is specifically called.
      SUCCEEDED: The upgrade Job is successfully completed.
      FAILED: The upgrade Job has failed. A failed job is resumable if the
        issues on the PC side are resolved. A job can also stay in failed
        state as its final state and a new job can be invoked.
      CANCELLED: The upgrade Job was cancelled. This will only happen when a
        upgrade is scraped after it is started, in instances like a newer
        version is available or customer criticality requires upgrade to be
        dropped for the time being. A new upgrade job to same or a different
        version should happen later.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLED = 5

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
    componentUpgrades = _messages.MessageField('VmwareComponentUpgrade', 1, repeated=True)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    name = _messages.StringField(4)
    progressPercent = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    startVersion = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    targetVersion = _messages.StringField(8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)
    upgradeType = _messages.EnumField('UpgradeTypeValueValuesEnum', 11)