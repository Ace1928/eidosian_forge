from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BootDiskDefaults(_messages.Message):
    """BootDiskDefaults hold information about the boot disk of a VM.

  Enums:
    DiskTypeValueValuesEnum: Optional. The type of disk provisioning to use
      for the VM.

  Fields:
    deviceName: Optional. Specifies a unique device name of your choice that
      is reflected into the /dev/disk/by-id/google-* tree of a Linux operating
      system running within the instance. If not specified, the server chooses
      a default device name to apply to this disk, in the form persistent-
      disk-x, where x is a number assigned by Google Compute Engine. This
      field is only applicable for persistent disks.
    diskName: Optional. The name of the disk.
    diskType: Optional. The type of disk provisioning to use for the VM.
    encryption: Optional. The encryption to apply to the boot disk.
    image: The image to use when creating the disk.
  """

    class DiskTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of disk provisioning to use for the VM.

    Values:
      COMPUTE_ENGINE_DISK_TYPE_UNSPECIFIED: An unspecified disk type. Will be
        used as STANDARD.
      COMPUTE_ENGINE_DISK_TYPE_STANDARD: A Standard disk type.
      COMPUTE_ENGINE_DISK_TYPE_SSD: SSD hard disk type.
      COMPUTE_ENGINE_DISK_TYPE_BALANCED: An alternative to SSD persistent
        disks that balance performance and cost.
    """
        COMPUTE_ENGINE_DISK_TYPE_UNSPECIFIED = 0
        COMPUTE_ENGINE_DISK_TYPE_STANDARD = 1
        COMPUTE_ENGINE_DISK_TYPE_SSD = 2
        COMPUTE_ENGINE_DISK_TYPE_BALANCED = 3
    deviceName = _messages.StringField(1)
    diskName = _messages.StringField(2)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 3)
    encryption = _messages.MessageField('Encryption', 4)
    image = _messages.MessageField('DiskImageDefaults', 5)