from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskImageTargetDetails(_messages.Message):
    """The target details of the image resource that will be created by the
  import job.

  Messages:
    LabelsValue: Optional. A map of labels to associate with the image.

  Fields:
    additionalLicenses: Optional. Additional licenses to assign to the image.
    dataDiskImageImport: Optional. Use to skip OS adaptation process.
    description: Optional. An optional description of the image.
    encryption: Immutable. The encryption to apply to the image.
    familyName: Optional. The name of the image family to which the new image
      belongs.
    imageName: Required. The name of the image to be created.
    labels: Optional. A map of labels to associate with the image.
    osAdaptationParameters: Optional. Use to set the parameters relevant for
      the OS adaptation process.
    singleRegionStorage: Optional. Set to true to set the image
      storageLocations to the single region of the import job. When false, the
      closest multi-region is selected.
    targetProject: Required. Reference to the TargetProject resource that
      represents the target project in which the imported image will be
      created.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A map of labels to associate with the image.

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
    additionalLicenses = _messages.StringField(1, repeated=True)
    dataDiskImageImport = _messages.MessageField('DataDiskImageImport', 2)
    description = _messages.StringField(3)
    encryption = _messages.MessageField('Encryption', 4)
    familyName = _messages.StringField(5)
    imageName = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    osAdaptationParameters = _messages.MessageField('ImageImportOsAdaptationParameters', 8)
    singleRegionStorage = _messages.BooleanField(9)
    targetProject = _messages.StringField(10)