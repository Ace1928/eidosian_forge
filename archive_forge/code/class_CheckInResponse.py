from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckInResponse(_messages.Message):
    """The response to the CheckIn method.

  Messages:
    FeaturesValue: Feature configuration for the operation.
    MetadataValue: The metadata that describes the operation assigned to the
      worker.

  Fields:
    deadline: The deadline by which the worker must request an extension. The
      backend will allow for network transmission time and other delays, but
      the worker must attempt to transmit the extension request no later than
      the deadline.
    features: Feature configuration for the operation.
    metadata: The metadata that describes the operation assigned to the
      worker.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FeaturesValue(_messages.Message):
        """Feature configuration for the operation.

    Messages:
      AdditionalProperty: An additional property for a FeaturesValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """The metadata that describes the operation assigned to the worker.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    deadline = _messages.StringField(1)
    features = _messages.MessageField('FeaturesValue', 2)
    metadata = _messages.MessageField('MetadataValue', 3)