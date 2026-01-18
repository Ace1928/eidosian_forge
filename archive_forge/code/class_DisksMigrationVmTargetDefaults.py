from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksMigrationVmTargetDefaults(_messages.Message):
    """Details for creation of a VM that migrated data disks will be attached
  to.

  Messages:
    LabelsValue: Optional. A map of labels to associate with the VM.
    MetadataValue: Optional. The metadata key/value pairs to assign to the VM.

  Fields:
    additionalLicenses: Optional. Additional licenses to assign to the VM.
    bootDiskDefaults: Optional. Details of the boot disk of the VM.
    computeScheduling: Optional. Compute instance scheduling information (if
      empty default is used).
    encryption: Optional. The encryption to apply to the VM.
    hostname: Optional. The hostname to assign to the VM.
    labels: Optional. A map of labels to associate with the VM.
    machineType: Required. The machine type to create the VM with.
    machineTypeSeries: Optional. The machine type series to create the VM
      with. For presentation only.
    metadata: Optional. The metadata key/value pairs to assign to the VM.
    networkInterfaces: Optional. NICs to attach to the VM.
    networkTags: Optional. A list of network tags to associate with the VM.
    secureBoot: Optional. Defines whether the instance has Secure Boot
      enabled. This can be set to true only if the VM boot option is EFI.
    serviceAccount: Optional. The service account to associate the VM with.
    vmName: Required. The name of the VM to create.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A map of labels to associate with the VM.

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
        """Optional. The metadata key/value pairs to assign to the VM.

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
    bootDiskDefaults = _messages.MessageField('BootDiskDefaults', 2)
    computeScheduling = _messages.MessageField('ComputeScheduling', 3)
    encryption = _messages.MessageField('Encryption', 4)
    hostname = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    machineType = _messages.StringField(7)
    machineTypeSeries = _messages.StringField(8)
    metadata = _messages.MessageField('MetadataValue', 9)
    networkInterfaces = _messages.MessageField('NetworkInterface', 10, repeated=True)
    networkTags = _messages.StringField(11, repeated=True)
    secureBoot = _messages.BooleanField(12)
    serviceAccount = _messages.StringField(13)
    vmName = _messages.StringField(14)