from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SavedAttachedDisk(_messages.Message):
    """DEPRECATED: Please use compute#savedDisk instead. An instance-attached
  disk resource.

  Enums:
    InterfaceValueValuesEnum: Specifies the disk interface to use for
      attaching this disk, which is either SCSI or NVME.
    ModeValueValuesEnum: The mode in which this disk is attached to the source
      instance, either READ_WRITE or READ_ONLY.
    StorageBytesStatusValueValuesEnum: [Output Only] An indicator whether
      storageBytes is in a stable state or it is being adjusted as a result of
      shared storage reallocation. This status can either be UPDATING, meaning
      the size of the snapshot is being updated, or UP_TO_DATE, meaning the
      size of the snapshot is up-to-date.
    TypeValueValuesEnum: Specifies the type of the attached disk, either
      SCRATCH or PERSISTENT.

  Fields:
    autoDelete: Specifies whether the disk will be auto-deleted when the
      instance is deleted (but not when the disk is detached from the
      instance).
    boot: Indicates that this is a boot disk. The virtual machine will use the
      first partition of the disk for its root filesystem.
    deviceName: Specifies the name of the disk attached to the source
      instance.
    diskEncryptionKey: The encryption key for the disk.
    diskSizeGb: The size of the disk in base-2 GB.
    diskType: [Output Only] URL of the disk type resource. For example:
      projects/project /zones/zone/diskTypes/pd-standard or pd-ssd
    guestOsFeatures: A list of features to enable on the guest operating
      system. Applicable only for bootable images. Read Enabling guest
      operating system features to see a list of available options.
    index: Specifies zero-based index of the disk that is attached to the
      source instance.
    interface: Specifies the disk interface to use for attaching this disk,
      which is either SCSI or NVME.
    kind: [Output Only] Type of the resource. Always compute#attachedDisk for
      attached disks.
    licenses: [Output Only] Any valid publicly visible licenses.
    mode: The mode in which this disk is attached to the source instance,
      either READ_WRITE or READ_ONLY.
    source: Specifies a URL of the disk attached to the source instance.
    storageBytes: [Output Only] A size of the storage used by the disk's
      snapshot by this machine image.
    storageBytesStatus: [Output Only] An indicator whether storageBytes is in
      a stable state or it is being adjusted as a result of shared storage
      reallocation. This status can either be UPDATING, meaning the size of
      the snapshot is being updated, or UP_TO_DATE, meaning the size of the
      snapshot is up-to-date.
    type: Specifies the type of the attached disk, either SCRATCH or
      PERSISTENT.
  """

    class InterfaceValueValuesEnum(_messages.Enum):
        """Specifies the disk interface to use for attaching this disk, which is
    either SCSI or NVME.

    Values:
      NVME: <no description>
      SCSI: <no description>
    """
        NVME = 0
        SCSI = 1

    class ModeValueValuesEnum(_messages.Enum):
        """The mode in which this disk is attached to the source instance, either
    READ_WRITE or READ_ONLY.

    Values:
      READ_ONLY: Attaches this disk in read-only mode. Multiple virtual
        machines can use a disk in read-only mode at a time.
      READ_WRITE: *[Default]* Attaches this disk in read-write mode. Only one
        virtual machine at a time can be attached to a disk in read-write
        mode.
    """
        READ_ONLY = 0
        READ_WRITE = 1

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

    class TypeValueValuesEnum(_messages.Enum):
        """Specifies the type of the attached disk, either SCRATCH or PERSISTENT.

    Values:
      PERSISTENT: <no description>
      SCRATCH: <no description>
    """
        PERSISTENT = 0
        SCRATCH = 1
    autoDelete = _messages.BooleanField(1)
    boot = _messages.BooleanField(2)
    deviceName = _messages.StringField(3)
    diskEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 4)
    diskSizeGb = _messages.IntegerField(5)
    diskType = _messages.StringField(6)
    guestOsFeatures = _messages.MessageField('GuestOsFeature', 7, repeated=True)
    index = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    interface = _messages.EnumField('InterfaceValueValuesEnum', 9)
    kind = _messages.StringField(10, default='compute#savedAttachedDisk')
    licenses = _messages.StringField(11, repeated=True)
    mode = _messages.EnumField('ModeValueValuesEnum', 12)
    source = _messages.StringField(13)
    storageBytes = _messages.IntegerField(14)
    storageBytesStatus = _messages.EnumField('StorageBytesStatusValueValuesEnum', 15)
    type = _messages.EnumField('TypeValueValuesEnum', 16)