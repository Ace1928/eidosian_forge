from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DimensionsInfo(_messages.Message):
    """The detailed quota information such as effective quota value for a
  combination of dimensions.

  Messages:
    DimensionsValue: The map of dimensions for this dimensions info. The key
      of a map entry is "region", "zone" or the name of a service specific
      dimension, and the value of a map entry is the value of the dimension.
      If a dimension does not appear in the map of dimensions, the dimensions
      info applies to all the dimension values except for those that have
      another DimenisonInfo instance configured for the specific value.
      Example: {"provider" : "Foo Inc"} where "provider" is a service specific
      dimension of a quota.

  Fields:
    applicableLocations: The applicable regions or zones of this dimensions
      info. The field will be set to ['global'] for quotas that are not per
      region or per zone. Otherwise, it will be set to the list of locations
      this dimension info is applicable to.
    details: Quota details for the specified dimensions.
    dimensions: The map of dimensions for this dimensions info. The key of a
      map entry is "region", "zone" or the name of a service specific
      dimension, and the value of a map entry is the value of the dimension.
      If a dimension does not appear in the map of dimensions, the dimensions
      info applies to all the dimension values except for those that have
      another DimenisonInfo instance configured for the specific value.
      Example: {"provider" : "Foo Inc"} where "provider" is a service specific
      dimension of a quota.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """The map of dimensions for this dimensions info. The key of a map entry
    is "region", "zone" or the name of a service specific dimension, and the
    value of a map entry is the value of the dimension. If a dimension does
    not appear in the map of dimensions, the dimensions info applies to all
    the dimension values except for those that have another DimenisonInfo
    instance configured for the specific value. Example: {"provider" : "Foo
    Inc"} where "provider" is a service specific dimension of a quota.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    applicableLocations = _messages.StringField(1, repeated=True)
    details = _messages.MessageField('QuotaDetails', 2)
    dimensions = _messages.MessageField('DimensionsValue', 3)