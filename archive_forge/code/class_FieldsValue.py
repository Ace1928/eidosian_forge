from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FieldsValue(_messages.Message):
    """Required. Map of tag template field IDs to the settings for the field.
    This map is an exhaustive list of the allowed fields. The map must contain
    at least one field and at most 500 fields. The keys to this map are tag
    template field IDs. The IDs have the following limitations: * Can contain
    uppercase and lowercase letters, numbers (0-9) and underscores (_). * Must
    be at least 1 character and at most 64 characters long. * Must start with
    a letter or underscore.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1TagTemplateField attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplateField', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)