from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MapValue(_messages.Message):
    """A map value.

  Messages:
    FieldsValue: The map's fields. The map keys represent field names. Field
      names matching the regular expression `__.*__` are reserved. Reserved
      field names are forbidden except in certain documented contexts. The map
      keys, represented as UTF-8, must not exceed 1,500 bytes and cannot be
      empty.

  Fields:
    fields: The map's fields. The map keys represent field names. Field names
      matching the regular expression `__.*__` are reserved. Reserved field
      names are forbidden except in certain documented contexts. The map keys,
      represented as UTF-8, must not exceed 1,500 bytes and cannot be empty.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FieldsValue(_messages.Message):
        """The map's fields. The map keys represent field names. Field names
    matching the regular expression `__.*__` are reserved. Reserved field
    names are forbidden except in certain documented contexts. The map keys,
    represented as UTF-8, must not exceed 1,500 bytes and cannot be empty.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A Value attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Value', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    fields = _messages.MessageField('FieldsValue', 1)