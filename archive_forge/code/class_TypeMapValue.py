from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TypeMapValue(_messages.Message):
    """A map from expression ids to types. Every expression node which has a
    type different than DYN has a mapping here. If an expression has type DYN,
    it is omitted from this map to save space.

    Messages:
      AdditionalProperty: An additional property for a TypeMapValue object.

    Fields:
      additionalProperties: Additional properties of type TypeMapValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TypeMapValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprType attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleApiExprType', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)