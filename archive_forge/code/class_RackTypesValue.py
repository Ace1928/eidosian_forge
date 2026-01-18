from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RackTypesValue(_messages.Message):
    """The map keyed by rack name and has value of RackType.

    Messages:
      AdditionalProperty: An additional property for a RackTypesValue object.

    Fields:
      additionalProperties: Additional properties of type RackTypesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RackTypesValue object.

      Enums:
        ValueValueValuesEnum:

      Fields:
        key: Name of the additional property.
        value: A ValueValueValuesEnum attribute.
      """

        class ValueValueValuesEnum(_messages.Enum):
            """ValueValueValuesEnum enum type.

        Values:
          RACK_TYPE_UNSPECIFIED: Unspecified rack type, single rack also
            belongs to this type.
          BASE: Base rack type, a pair of two modified Config-1 racks
            containing Aggregation switches.
          EXPANSION: Expansion rack type, also known as standalone racks,
            added by customers on demand.
        """
            RACK_TYPE_UNSPECIFIED = 0
            BASE = 1
            EXPANSION = 2
        key = _messages.StringField(1)
        value = _messages.EnumField('ValueValueValuesEnum', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)