from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsVmDetails(_messages.Message):
    """AwsVmDetails describes a VM in AWS.

  Enums:
    ArchitectureValueValuesEnum: The CPU architecture.
    BootOptionValueValuesEnum: The VM Boot Option.
    PowerStateValueValuesEnum: Output only. The power state of the VM at the
      moment list was taken.
    VirtualizationTypeValueValuesEnum: The virtualization type.

  Messages:
    TagsValue: The tags of the VM.

  Fields:
    architecture: The CPU architecture.
    bootOption: The VM Boot Option.
    committedStorageMb: The total size of the storage allocated to the VM in
      MB.
    cpuCount: The number of cpus the VM has.
    diskCount: The number of disks the VM has.
    displayName: The display name of the VM. Note that this value is not
      necessarily unique.
    instanceType: The instance type of the VM.
    memoryMb: The memory size of the VM in MB.
    osDescription: The VM's OS.
    powerState: Output only. The power state of the VM at the moment list was
      taken.
    securityGroups: The security groups the VM belongs to.
    sourceDescription: The descriptive name of the AWS's source this VM is
      connected to.
    sourceId: The id of the AWS's source this VM is connected to.
    tags: The tags of the VM.
    virtualizationType: The virtualization type.
    vmId: The VM ID in AWS.
    vpcId: The VPC ID the VM belongs to.
    zone: The AWS zone of the VM.
  """

    class ArchitectureValueValuesEnum(_messages.Enum):
        """The CPU architecture.

    Values:
      VM_ARCHITECTURE_UNSPECIFIED: The architecture is unknown.
      I386: The architecture is I386.
      X86_64: The architecture is X86_64.
      ARM64: The architecture is ARM64.
      X86_64_MAC: The architecture is X86_64_MAC.
    """
        VM_ARCHITECTURE_UNSPECIFIED = 0
        I386 = 1
        X86_64 = 2
        ARM64 = 3
        X86_64_MAC = 4

    class BootOptionValueValuesEnum(_messages.Enum):
        """The VM Boot Option.

    Values:
      BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      EFI: The boot option is UEFI.
      BIOS: The boot option is LEGACY-BIOS.
    """
        BOOT_OPTION_UNSPECIFIED = 0
        EFI = 1
        BIOS = 2

    class PowerStateValueValuesEnum(_messages.Enum):
        """Output only. The power state of the VM at the moment list was taken.

    Values:
      POWER_STATE_UNSPECIFIED: Power state is not specified.
      ON: The VM is turned on.
      OFF: The VM is turned off.
      SUSPENDED: The VM is suspended. This is similar to hibernation or sleep
        mode.
      PENDING: The VM is starting.
    """
        POWER_STATE_UNSPECIFIED = 0
        ON = 1
        OFF = 2
        SUSPENDED = 3
        PENDING = 4

    class VirtualizationTypeValueValuesEnum(_messages.Enum):
        """The virtualization type.

    Values:
      VM_VIRTUALIZATION_TYPE_UNSPECIFIED: The virtualization type is unknown.
      HVM: The virtualziation type is HVM.
      PARAVIRTUAL: The virtualziation type is PARAVIRTUAL.
    """
        VM_VIRTUALIZATION_TYPE_UNSPECIFIED = 0
        HVM = 1
        PARAVIRTUAL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """The tags of the VM.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    architecture = _messages.EnumField('ArchitectureValueValuesEnum', 1)
    bootOption = _messages.EnumField('BootOptionValueValuesEnum', 2)
    committedStorageMb = _messages.IntegerField(3)
    cpuCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    diskCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    displayName = _messages.StringField(6)
    instanceType = _messages.StringField(7)
    memoryMb = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    osDescription = _messages.StringField(9)
    powerState = _messages.EnumField('PowerStateValueValuesEnum', 10)
    securityGroups = _messages.MessageField('AwsSecurityGroup', 11, repeated=True)
    sourceDescription = _messages.StringField(12)
    sourceId = _messages.StringField(13)
    tags = _messages.MessageField('TagsValue', 14)
    virtualizationType = _messages.EnumField('VirtualizationTypeValueValuesEnum', 15)
    vmId = _messages.StringField(16)
    vpcId = _messages.StringField(17)
    zone = _messages.StringField(18)