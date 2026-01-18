from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MetadataConfigsValue(_messages.Message):
    """Mapping of field name to its configuration.

    Messages:
      AdditionalProperty: An additional property for a MetadataConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MetadataConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MetadataConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A MetadataConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('MetadataConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)