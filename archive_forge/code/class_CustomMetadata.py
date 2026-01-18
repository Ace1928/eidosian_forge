from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomMetadata(_messages.Message):
    """CustomMetadata contains information from a user-defined operation.

  Messages:
    ValuesValue: Output only. Key-value pairs provided by the user-defined
      operation.

  Fields:
    values: Output only. Key-value pairs provided by the user-defined
      operation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ValuesValue(_messages.Message):
        """Output only. Key-value pairs provided by the user-defined operation.

    Messages:
      AdditionalProperty: An additional property for a ValuesValue object.

    Fields:
      additionalProperties: Additional properties of type ValuesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    values = _messages.MessageField('ValuesValue', 1)