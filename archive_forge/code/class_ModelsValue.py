from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ModelsValue(_messages.Message):
    """Map of locale (language code) -> models

    Messages:
      AdditionalProperty: An additional property for a ModelsValue object.

    Fields:
      additionalProperties: Additional properties of type ModelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ModelsValue object.

      Fields:
        key: Name of the additional property.
        value: A ModelMetadata attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ModelMetadata', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)