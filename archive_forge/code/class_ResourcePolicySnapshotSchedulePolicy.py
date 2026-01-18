from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicySnapshotSchedulePolicy(_messages.Message):
    """A snapshot schedule policy specifies when and how frequently snapshots
  are to be created for the target disk. Also specifies how many and how long
  these scheduled snapshots should be retained.

  Fields:
    retentionPolicy: Retention policy applied to snapshots created by this
      resource policy.
    schedule: A Vm Maintenance Policy specifies what kind of infrastructure
      maintenance we are allowed to perform on this VM and when. Schedule that
      is applied to disks covered by this policy.
    snapshotProperties: Properties with which snapshots are created such as
      labels, encryption keys.
  """
    retentionPolicy = _messages.MessageField('ResourcePolicySnapshotSchedulePolicyRetentionPolicy', 1)
    schedule = _messages.MessageField('ResourcePolicySnapshotSchedulePolicySchedule', 2)
    snapshotProperties = _messages.MessageField('ResourcePolicySnapshotSchedulePolicySnapshotProperties', 3)