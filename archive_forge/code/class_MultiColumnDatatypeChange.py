from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiColumnDatatypeChange(_messages.Message):
    """Options to configure rule type MultiColumnDatatypeChange. The rule is
  used to change the data type and associated properties of multiple columns
  at once. The rule filter field can refer to one or more entities. The rule
  scope can be one of:Column. This rule requires additional filters to be
  specified beyond the basic rule filter field, which is the source data type,
  but the rule supports additional filtering capabilities such as the minimum
  and maximum field length. All additional filters which are specified are
  required to be met in order for the rule to be applied (logical AND between
  the fields).

  Messages:
    CustomFeaturesValue: Optional. Custom engine specific features.

  Fields:
    customFeatures: Optional. Custom engine specific features.
    newDataType: Required. New data type.
    overrideFractionalSecondsPrecision: Optional. Column fractional seconds
      precision - used only for timestamp based datatypes - if not specified
      and relevant uses the source column fractional seconds precision.
    overrideLength: Optional. Column length - e.g. varchar (50) - if not
      specified and relevant uses the source column length.
    overridePrecision: Optional. Column precision - when relevant - if not
      specified and relevant uses the source column precision.
    overrideScale: Optional. Column scale - when relevant - if not specified
      and relevant uses the source column scale.
    sourceDataTypeFilter: Required. Filter on source data type.
    sourceNumericFilter: Optional. Filter for fixed point number data types
      such as NUMERIC/NUMBER.
    sourceTextFilter: Optional. Filter for text-based data types like varchar.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Optional. Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 1)
    newDataType = _messages.StringField(2)
    overrideFractionalSecondsPrecision = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    overrideLength = _messages.IntegerField(4)
    overridePrecision = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    overrideScale = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    sourceDataTypeFilter = _messages.StringField(7)
    sourceNumericFilter = _messages.MessageField('SourceNumericFilter', 8)
    sourceTextFilter = _messages.MessageField('SourceTextFilter', 9)