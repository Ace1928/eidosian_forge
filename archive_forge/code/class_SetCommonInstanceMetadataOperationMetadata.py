from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetCommonInstanceMetadataOperationMetadata(_messages.Message):
    """A SetCommonInstanceMetadataOperationMetadata object.

  Messages:
    PerLocationOperationsValue: [Output Only] Status information per location
      (location name is key). Example key: zones/us-central1-a

  Fields:
    clientOperationId: [Output Only] The client operation id.
    perLocationOperations: [Output Only] Status information per location
      (location name is key). Example key: zones/us-central1-a
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PerLocationOperationsValue(_messages.Message):
        """[Output Only] Status information per location (location name is key).
    Example key: zones/us-central1-a

    Messages:
      AdditionalProperty: An additional property for a
        PerLocationOperationsValue object.

    Fields:
      additionalProperties: Additional properties of type
        PerLocationOperationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PerLocationOperationsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          SetCommonInstanceMetadataOperationMetadataPerLocationOperationInfo
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('SetCommonInstanceMetadataOperationMetadataPerLocationOperationInfo', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    clientOperationId = _messages.StringField(1)
    perLocationOperations = _messages.MessageField('PerLocationOperationsValue', 2)