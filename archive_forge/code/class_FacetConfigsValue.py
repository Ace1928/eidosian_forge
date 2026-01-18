from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FacetConfigsValue(_messages.Message):
    """A map between facet name and its configuration within this catalog.

    Messages:
      AdditionalProperty: An additional property for a FacetConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type FacetConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FacetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A FacetConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('FacetConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)