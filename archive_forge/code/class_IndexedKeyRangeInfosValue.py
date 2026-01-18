from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class IndexedKeyRangeInfosValue(_messages.Message):
    """The (sparse) mapping from time interval index to an
    IndexedKeyRangeInfos message, representing those time intervals for which
    there are informational messages concerning key ranges.

    Messages:
      AdditionalProperty: An additional property for a
        IndexedKeyRangeInfosValue object.

    Fields:
      additionalProperties: Additional properties of type
        IndexedKeyRangeInfosValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a IndexedKeyRangeInfosValue object.

      Fields:
        key: Name of the additional property.
        value: A IndexedKeyRangeInfos attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('IndexedKeyRangeInfos', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)