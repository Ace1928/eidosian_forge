from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeEngineDisksTargetDefaults(_messages.Message):
    """ComputeEngineDisksTargetDefaults is a collection of details for creating
  Persistent Disks in a target Compute Engine project.

  Fields:
    disks: The details of each Persistent Disk to create.
    disksTargetDefaults: Details of the disk only migration target.
    targetProject: The full path of the resource of type TargetProject which
      represents the Compute Engine project in which to create the Persistent
      Disks.
    vmTargetDefaults: Details of the VM migration target.
    zone: The zone in which to create the Persistent Disks.
  """
    disks = _messages.MessageField('PersistentDiskDefaults', 1, repeated=True)
    disksTargetDefaults = _messages.MessageField('DisksMigrationDisksTargetDefaults', 2)
    targetProject = _messages.StringField(3)
    vmTargetDefaults = _messages.MessageField('DisksMigrationVmTargetDefaults', 4)
    zone = _messages.StringField(5)