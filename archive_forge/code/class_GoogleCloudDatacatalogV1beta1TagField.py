from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1TagField(_messages.Message):
    """Contains the value and supporting information for a field within a Tag.

  Fields:
    boolValue: Holds the value for a tag field with boolean type.
    displayName: Output only. The display name of this field.
    doubleValue: Holds the value for a tag field with double type.
    enumValue: Holds the value for a tag field with enum type. This value must
      be one of the allowed values in the definition of this enum.
    order: Output only. The order of this field with respect to other fields
      in this tag. It can be set in Tag. For example, a higher value can
      indicate a more important field. The value can be negative. Multiple
      fields can have the same order, and field orders within a tag do not
      have to be sequential.
    stringValue: Holds the value for a tag field with string type.
    timestampValue: Holds the value for a tag field with timestamp type.
  """
    boolValue = _messages.BooleanField(1)
    displayName = _messages.StringField(2)
    doubleValue = _messages.FloatField(3)
    enumValue = _messages.MessageField('GoogleCloudDatacatalogV1beta1TagFieldEnumValue', 4)
    order = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    stringValue = _messages.StringField(6)
    timestampValue = _messages.StringField(7)