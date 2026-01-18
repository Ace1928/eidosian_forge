from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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
        value: A AttributeValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AttributeValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)