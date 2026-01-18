from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeEngineDisksTargetDetails(_messages.Message):
    """ComputeEngineDisksTargetDetails is a collection of created Persistent
  Disks details.

  Fields:
    disks: The details of each created Persistent Disk.
    disksTargetDetails: Details of the disks-only migration target.
    vmTargetDetails: Details for the VM the migrated data disks are attached
      to.
  """
    disks = _messages.MessageField('PersistentDisk', 1, repeated=True)
    disksTargetDetails = _messages.MessageField('DisksMigrationDisksTargetDetails', 2)
    vmTargetDetails = _messages.MessageField('DisksMigrationVmTargetDetails', 3)