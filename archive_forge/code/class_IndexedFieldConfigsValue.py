from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class IndexedFieldConfigsValue(_messages.Message):
    """A map between user-defined key and IndexedFieldConfig, where the key
    is indexed with the value configured by IndexedFieldConfig and customers
    can use the key as search operator if the expression is nonempty. If the
    expression in CatalogIndexedFieldConfig is empty, the key must be "".

    Messages:
      AdditionalProperty: An additional property for a
        IndexedFieldConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        IndexedFieldConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a IndexedFieldConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A CatalogIndexedFieldConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('CatalogIndexedFieldConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)