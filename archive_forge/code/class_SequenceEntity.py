from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SequenceEntity(_messages.Message):
    """Sequence's parent is a schema.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    cache: Indicates number of entries to cache / precreate.
    customFeatures: Custom engine specific features.
    cycle: Indicates whether the sequence value should cycle through.
    increment: Increment value for the sequence.
    maxValue: Maximum number for the sequence represented as bytes to
      accommodate large. numbers
    minValue: Minimum number for the sequence represented as bytes to
      accommodate large. numbers
    startValue: Start number for the sequence represented as bytes to
      accommodate large. numbers
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cache = _messages.IntegerField(1)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 2)
    cycle = _messages.BooleanField(3)
    increment = _messages.IntegerField(4)
    maxValue = _messages.BytesField(5)
    minValue = _messages.BytesField(6)
    startValue = _messages.BytesField(7)