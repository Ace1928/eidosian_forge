from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SavedDisk(_messages.Message):
    """An instance-attached disk resource.

  Enums:
    ArchitectureValueValuesEnum: [Output Only] The architecture of the
      attached disk.
    StorageBytesStatusValueValuesEnum: [Output Only] An indicator whether
      storageBytes is in a stable state or it is being adjusted as a result of
      shared storage reallocation. This status can either be UPDATING, meaning
      the size of the snapshot is being updated, or UP_TO_DATE, meaning the
      size of the snapshot is up-to-date.

  Fields:
    architecture: [Output Only] The architecture of the attached disk.
    kind: [Output Only] Type of the resource. Always compute#savedDisk for
      attached disks.
    sourceDisk: Specifies a URL of the disk attached to the source instance.
    storageBytes: [Output Only] Size of the individual disk snapshot used by
      this machine image.
    storageBytesStatus: [Output Only] An indicator whether storageBytes is in
      a stable state or it is being adjusted as a result of shared storage
      reallocation. This status can either be UPDATING, meaning the size of
      the snapshot is being updated, or UP_TO_DATE, meaning the size of the
      snapshot is up-to-date.
  """

    class ArchitectureValueValuesEnum(_messages.Enum):
        """[Output Only] The architecture of the attached disk.

    Values:
      ARCHITECTURE_UNSPECIFIED: Default value indicating Architecture is not
        set.
      ARM64: Machines with architecture ARM64
      X86_64: Machines with architecture X86_64
    """
        ARCHITECTURE_UNSPECIFIED = 0
        ARM64 = 1
        X86_64 = 2

    class StorageBytesStatusValueValuesEnum(_messages.Enum):
        """[Output Only] An indicator whether storageBytes is in a stable state
    or it is being adjusted as a result of shared storage reallocation. This
    status can either be UPDATING, meaning the size of the snapshot is being
    updated, or UP_TO_DATE, meaning the size of the snapshot is up-to-date.

    Values:
      UPDATING: <no description>
      UP_TO_DATE: <no description>
    """
        UPDATING = 0
        UP_TO_DATE = 1
    architecture = _messages.EnumField('ArchitectureValueValuesEnum', 1)
    kind = _messages.StringField(2, default='compute#savedDisk')
    sourceDisk = _messages.StringField(3)
    storageBytes = _messages.IntegerField(4)
    storageBytesStatus = _messages.EnumField('StorageBytesStatusValueValuesEnum', 5)