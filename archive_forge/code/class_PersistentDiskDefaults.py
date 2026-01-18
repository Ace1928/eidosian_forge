from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistentDiskDefaults(_messages.Message):
    """Details for creation of a Persistent Disk.

  Enums:
    DiskTypeValueValuesEnum: The disk type to use.

  Messages:
    AdditionalLabelsValue: A map of labels to associate with the Persistent
      Disk.

  Fields:
    additionalLabels: A map of labels to associate with the Persistent Disk.
    diskName: Optional. The name of the Persistent Disk to create.
    diskType: The disk type to use.
    encryption: Optional. The encryption to apply to the disk.
    sourceDiskNumber: Required. The ordinal number of the source VM disk.
    vmAttachmentDetails: Optional. Details for attachment of the disk to a VM.
      Used when the disk is set to be attacked to a target VM.
  """

    class DiskTypeValueValuesEnum(_messages.Enum):
        """The disk type to use.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AdditionalLabelsValue(_messages.Message):
        """A map of labels to associate with the Persistent Disk.

    Messages:
      AdditionalProperty: An additional property for a AdditionalLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AdditionalLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AdditionalLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalLabels = _messages.MessageField('AdditionalLabelsValue', 1)
    diskName = _messages.StringField(2)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 3)
    encryption = _messages.MessageField('Encryption', 4)
    sourceDiskNumber = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    vmAttachmentDetails = _messages.MessageField('VmAttachmentDetails', 6)