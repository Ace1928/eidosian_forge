from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdditionalDetailsValue(_messages.Message):
    """The following map contains fields that are not explicitly mentioned
    above,this give connectors the flexibility to add new metadata fields.

    Messages:
      AdditionalProperty: An additional property for a AdditionalDetailsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AdditionalDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)