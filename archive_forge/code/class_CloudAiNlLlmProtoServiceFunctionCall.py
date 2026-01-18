from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceFunctionCall(_messages.Message):
    """Function call details.

  Messages:
    ArgsValue: The function parameters and values in JSON format.

  Fields:
    args: The function parameters and values in JSON format.
    name: Required. The name of the function to call.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgsValue(_messages.Message):
        """The function parameters and values in JSON format.

    Messages:
      AdditionalProperty: An additional property for a ArgsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    args = _messages.MessageField('ArgsValue', 1)
    name = _messages.StringField(2)