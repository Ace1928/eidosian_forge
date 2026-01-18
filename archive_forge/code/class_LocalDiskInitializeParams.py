from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalDiskInitializeParams(_messages.Message):
    """Input only. Specifies the parameters for a new disk that will be created
  alongside the new instance. Use initialization parameters to create boot
  disks or local SSDs attached to the new runtime. This property is mutually
  exclusive with the source property; you can only define one or the other,
  but not both.

  Enums:
    DiskTypeValueValuesEnum: Input only. The type of the boot disk attached to
      this instance, defaults to standard persistent disk (`PD_STANDARD`).

  Messages:
    LabelsValue: Optional. Labels to apply to this disk. These can be later
      modified by the disks.setLabels method. This field is only applicable
      for persistent disks.

  Fields:
    description: Optional. Provide this property when creating the disk.
    diskName: Optional. Specifies the disk name. If not specified, the default
      is to use the name of the instance. If the disk with the instance name
      exists already in the given zone/region, a new name will be
      automatically generated.
    diskSizeGb: Optional. Specifies the size of the disk in base-2 GB. If not
      specified, the disk will be the same size as the image (usually 10GB).
      If specified, the size must be equal to or larger than 10GB. Default 100
      GB.
    diskType: Input only. The type of the boot disk attached to this instance,
      defaults to standard persistent disk (`PD_STANDARD`).
    labels: Optional. Labels to apply to this disk. These can be later
      modified by the disks.setLabels method. This field is only applicable
      for persistent disks.
  """

    class DiskTypeValueValuesEnum(_messages.Enum):
        """Input only. The type of the boot disk attached to this instance,
    defaults to standard persistent disk (`PD_STANDARD`).

    Values:
      DISK_TYPE_UNSPECIFIED: Disk type not set.
      PD_STANDARD: Standard persistent disk type.
      PD_SSD: SSD persistent disk type.
      PD_BALANCED: Balanced persistent disk type.
      PD_EXTREME: Extreme persistent disk type.
    """
        DISK_TYPE_UNSPECIFIED = 0
        PD_STANDARD = 1
        PD_SSD = 2
        PD_BALANCED = 3
        PD_EXTREME = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels to apply to this disk. These can be later modified by
    the disks.setLabels method. This field is only applicable for persistent
    disks.

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
    description = _messages.StringField(1)
    diskName = _messages.StringField(2)
    diskSizeGb = _messages.IntegerField(3)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 4)
    labels = _messages.MessageField('LabelsValue', 5)