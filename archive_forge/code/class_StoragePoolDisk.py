from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragePoolDisk(_messages.Message):
    """A StoragePoolDisk object.

  Enums:
    StatusValueValuesEnum: [Output Only] The disk status.

  Fields:
    attachedInstances: [Output Only] Instances this disk is attached to.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    disk: [Output Only] The URL of the disk.
    name: [Output Only] The name of the disk.
    provisionedIops: [Output Only] The number of IOPS provisioned for the
      disk.
    provisionedThroughput: [Output Only] The throughput provisioned for the
      disk.
    resourcePolicies: [Output Only] Resource policies applied to disk for
      automatic snapshot creations.
    sizeGb: [Output Only] The disk size, in GB.
    status: [Output Only] The disk status.
    type: [Output Only] The disk type.
    usedBytes: [Output Only] Amount of disk space used.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The disk status.

    Values:
      CREATING: Disk is provisioning
      DELETING: Disk is deleting.
      FAILED: Disk creation failed.
      READY: Disk is ready for use.
      RESTORING: Source data is being copied into the disk.
      UNAVAILABLE: Disk is currently unavailable and cannot be accessed,
        attached or detached.
    """
        CREATING = 0
        DELETING = 1
        FAILED = 2
        READY = 3
        RESTORING = 4
        UNAVAILABLE = 5
    attachedInstances = _messages.StringField(1, repeated=True)
    creationTimestamp = _messages.StringField(2)
    disk = _messages.StringField(3)
    name = _messages.StringField(4)
    provisionedIops = _messages.IntegerField(5)
    provisionedThroughput = _messages.IntegerField(6)
    resourcePolicies = _messages.StringField(7, repeated=True)
    sizeGb = _messages.IntegerField(8)
    status = _messages.EnumField('StatusValueValuesEnum', 9)
    type = _messages.StringField(10)
    usedBytes = _messages.IntegerField(11)