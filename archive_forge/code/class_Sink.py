from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Sink(_messages.Message):
    """A sink that records can be encoded and written to.

  Messages:
    CodecValue: The codec to use to encode data written to the sink.
    SpecValue: The sink to write to, plus its parameters.

  Fields:
    codec: The codec to use to encode data written to the sink.
    spec: The sink to write to, plus its parameters.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CodecValue(_messages.Message):
        """The codec to use to encode data written to the sink.

    Messages:
      AdditionalProperty: An additional property for a CodecValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CodecValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SpecValue(_messages.Message):
        """The sink to write to, plus its parameters.

    Messages:
      AdditionalProperty: An additional property for a SpecValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SpecValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    codec = _messages.MessageField('CodecValue', 1)
    spec = _messages.MessageField('SpecValue', 2)