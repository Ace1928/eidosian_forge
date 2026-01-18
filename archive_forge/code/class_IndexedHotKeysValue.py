from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class IndexedHotKeysValue(_messages.Message):
    """The (sparse) mapping from time index to an IndexedHotKey message,
    representing those time intervals for which there are hot keys.

    Messages:
      AdditionalProperty: An additional property for a IndexedHotKeysValue
        object.

    Fields:
      additionalProperties: Additional properties of type IndexedHotKeysValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a IndexedHotKeysValue object.

      Fields:
        key: Name of the additional property.
        value: A IndexedHotKey attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('IndexedHotKey', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)