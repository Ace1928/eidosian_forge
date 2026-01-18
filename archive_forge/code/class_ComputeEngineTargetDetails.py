from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeEngineTargetDetails(_messages.Message):
    """ComputeEngineTargetDetails is a collection of details for creating a VM
  in a target Compute Engine project.

  Enums:
    BootOptionValueValuesEnum: The VM Boot Option, as set in the source VM.
    DiskTypeValueValuesEnum: The disk type to use in the VM.
    LicenseTypeValueValuesEnum: The license type to use in OS adaptation.

  Messages:
    LabelsValue: A map of labels to associate with the VM.
    MetadataValue: The metadata key/value pairs to assign to the VM.

  Fields:
    additionalLicenses: Additional licenses to assign to the VM.
    appliedLicense: The OS license returned from the adaptation module report.
    bootOption: The VM Boot Option, as set in the source VM.
    computeScheduling: Compute instance scheduling information (if empty
      default is used).
    diskType: The disk type to use in the VM.
    encryption: Optional. The encryption to apply to the VM disks.
    hostname: The hostname to assign to the VM.
    labels: A map of labels to associate with the VM.
    licenseType: The license type to use in OS adaptation.
    machineType: The machine type to create the VM with.
    machineTypeSeries: The machine type series to create the VM with.
    metadata: The metadata key/value pairs to assign to the VM.
    networkInterfaces: List of NICs connected to this VM.
    networkTags: A list of network tags to associate with the VM.
    project: The Google Cloud target project ID or project name.
    secureBoot: Defines whether the instance has Secure Boot enabled. This can
      be set to true only if the VM boot option is EFI.
    serviceAccount: The service account to associate the VM with.
    vmName: The name of the VM to create.
    zone: The zone in which to create the VM.
  """

    class BootOptionValueValuesEnum(_messages.Enum):
        """The VM Boot Option, as set in the source VM.

    Values:
      COMPUTE_ENGINE_BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      COMPUTE_ENGINE_BOOT_OPTION_EFI: The boot option is EFI.
      COMPUTE_ENGINE_BOOT_OPTION_BIOS: The boot option is BIOS.
    """
        COMPUTE_ENGINE_BOOT_OPTION_UNSPECIFIED = 0
        COMPUTE_ENGINE_BOOT_OPTION_EFI = 1
        COMPUTE_ENGINE_BOOT_OPTION_BIOS = 2

    class DiskTypeValueValuesEnum(_messages.Enum):
        """The disk type to use in the VM.

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

    class LicenseTypeValueValuesEnum(_messages.Enum):
        """The license type to use in OS adaptation.

    Values:
      COMPUTE_ENGINE_LICENSE_TYPE_DEFAULT: The license type is the default for
        the OS.
      COMPUTE_ENGINE_LICENSE_TYPE_PAYG: The license type is Pay As You Go
        license type.
      COMPUTE_ENGINE_LICENSE_TYPE_BYOL: The license type is Bring Your Own
        License type.
    """
        COMPUTE_ENGINE_LICENSE_TYPE_DEFAULT = 0
        COMPUTE_ENGINE_LICENSE_TYPE_PAYG = 1
        COMPUTE_ENGINE_LICENSE_TYPE_BYOL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A map of labels to associate with the VM.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """The metadata key/value pairs to assign to the VM.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalLicenses = _messages.StringField(1, repeated=True)
    appliedLicense = _messages.MessageField('AppliedLicense', 2)
    bootOption = _messages.EnumField('BootOptionValueValuesEnum', 3)
    computeScheduling = _messages.MessageField('ComputeScheduling', 4)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 5)
    encryption = _messages.MessageField('Encryption', 6)
    hostname = _messages.StringField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    licenseType = _messages.EnumField('LicenseTypeValueValuesEnum', 9)
    machineType = _messages.StringField(10)
    machineTypeSeries = _messages.StringField(11)
    metadata = _messages.MessageField('MetadataValue', 12)
    networkInterfaces = _messages.MessageField('NetworkInterface', 13, repeated=True)
    networkTags = _messages.StringField(14, repeated=True)
    project = _messages.StringField(15)
    secureBoot = _messages.BooleanField(16)
    serviceAccount = _messages.StringField(17)
    vmName = _messages.StringField(18)
    zone = _messages.StringField(19)