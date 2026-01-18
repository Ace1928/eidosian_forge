from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class GoldenInfoTypeMappingValue(_messages.Message):
    """Optional. Similar to `eval_info_type_mapping`, infoType mapping for
    `golden_store`.

    Messages:
      AdditionalProperty: An additional property for a
        GoldenInfoTypeMappingValue object.

    Fields:
      additionalProperties: Additional properties of type
        GoldenInfoTypeMappingValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a GoldenInfoTypeMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)