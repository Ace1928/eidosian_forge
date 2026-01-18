from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1TagField(_messages.Message):
    """Contains the value and additional information on a field within a Tag.

  Fields:
    boolValue: The value of a tag field with a boolean type.
    displayName: Output only. The display name of this field.
    doubleValue: The value of a tag field with a double type.
    enumValue: The value of a tag field with an enum type. This value must be
      one of the allowed values listed in this enum.
    order: Output only. The order of this field with respect to other fields
      in this tag. Can be set by Tag. For example, a higher value can indicate
      a more important field. The value can be negative. Multiple fields can
      have the same order, and field orders within a tag don't have to be
      sequential.
    richtextValue: The value of a tag field with a rich text type. The maximum
      length is 10 MiB as this value holds HTML descriptions including encoded
      images. The maximum length of the text without images is 100 KiB.
    stringValue: The value of a tag field with a string type. The maximum
      length is 2000 UTF-8 characters.
    timestampValue: The value of a tag field with a timestamp type.
  """
    boolValue = _messages.BooleanField(1)
    displayName = _messages.StringField(2)
    doubleValue = _messages.FloatField(3)
    enumValue = _messages.MessageField('GoogleCloudDatacatalogV1TagFieldEnumValue', 4)
    order = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    richtextValue = _messages.StringField(6)
    stringValue = _messages.StringField(7)
    timestampValue = _messages.StringField(8)