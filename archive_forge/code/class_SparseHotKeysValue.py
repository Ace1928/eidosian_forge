from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SparseHotKeysValue(_messages.Message):
    """A (sparse) mapping from key bucket index to the index of the specific
    hot row key for that key bucket. The index of the hot row key can be
    translated to the actual row key via the
    ScanData.VisualizationData.indexed_keys repeated field.

    Messages:
      AdditionalProperty: An additional property for a SparseHotKeysValue
        object.

    Fields:
      additionalProperties: Additional properties of type SparseHotKeysValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SparseHotKeysValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)