from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1Attributes(_messages.Message):
    """A set of attributes, each in the format `[KEY]:[VALUE]`.

  Messages:
    AttributeMapValue: The set of attributes. Each attribute's key can be up
      to 128 bytes long. The value can be a string up to 256 bytes, a signed
      64-bit integer, or the Boolean values `true` and `false`. For example:
      "/instance_id": "my-instance" "/http/user_agent": ""
      "/http/request_bytes": 300 "abc.com/myattribute": true

  Fields:
    attributeMap: The set of attributes. Each attribute's key can be up to 128
      bytes long. The value can be a string up to 256 bytes, a signed 64-bit
      integer, or the Boolean values `true` and `false`. For example:
      "/instance_id": "my-instance" "/http/user_agent": ""
      "/http/request_bytes": 300 "abc.com/myattribute": true
    droppedAttributesCount: The number of attributes that were discarded.
      Attributes can be discarded because their keys are too long or because
      there are too many attributes. If this value is 0 then all attributes
      are valid.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributeMapValue(_messages.Message):
        """The set of attributes. Each attribute's key can be up to 128 bytes
    long. The value can be a string up to 256 bytes, a signed 64-bit integer,
    or the Boolean values `true` and `false`. For example: "/instance_id":
    "my-instance" "/http/user_agent": "" "/http/request_bytes": 300
    "abc.com/myattribute": true

    Messages:
      AdditionalProperty: An additional property for a AttributeMapValue
        object.

    Fields:
      additionalProperties: Additional properties of type AttributeMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributeMapValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiServicecontrolV1AttributeValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleApiServicecontrolV1AttributeValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributeMap = _messages.MessageField('AttributeMapValue', 1)
    droppedAttributesCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)