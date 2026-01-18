from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemVariables(_messages.Message):
    """System variables given to a query.

  Messages:
    TypesValue: Output only. Data type for each system variable.
    ValuesValue: Output only. Value for each system variable.

  Fields:
    types: Output only. Data type for each system variable.
    values: Output only. Value for each system variable.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TypesValue(_messages.Message):
        """Output only. Data type for each system variable.

    Messages:
      AdditionalProperty: An additional property for a TypesValue object.

    Fields:
      additionalProperties: Additional properties of type TypesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TypesValue object.

      Fields:
        key: Name of the additional property.
        value: A StandardSqlDataType attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StandardSqlDataType', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ValuesValue(_messages.Message):
        """Output only. Value for each system variable.

    Messages:
      AdditionalProperty: An additional property for a ValuesValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    types = _messages.MessageField('TypesValue', 1)
    values = _messages.MessageField('ValuesValue', 2)