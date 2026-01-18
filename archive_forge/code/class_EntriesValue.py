from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EntriesValue(_messages.Message):
    """Map of a partner metadata that belong to the same subdomain. It
    accepts any value including google.protobuf.Struct.

    Messages:
      AdditionalProperty: An additional property for a EntriesValue object.

    Fields:
      additionalProperties: Additional properties of type EntriesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EntriesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)