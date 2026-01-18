from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancePropertiesPatch(_messages.Message):
    """Represents the change that you want to make to the instance properties.

  Messages:
    LabelsValue: The label key-value pairs that you want to patch onto the
      instance.
    MetadataValue: The metadata key-value pairs that you want to patch onto
      the instance. For more information, see Project and instance metadata.

  Fields:
    labels: The label key-value pairs that you want to patch onto the
      instance.
    metadata: The metadata key-value pairs that you want to patch onto the
      instance. For more information, see Project and instance metadata.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The label key-value pairs that you want to patch onto the instance.

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
        """The metadata key-value pairs that you want to patch onto the instance.
    For more information, see Project and instance metadata.

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
    labels = _messages.MessageField('LabelsValue', 1)
    metadata = _messages.MessageField('MetadataValue', 2)