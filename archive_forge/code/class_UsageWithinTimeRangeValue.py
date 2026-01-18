from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class UsageWithinTimeRangeValue(_messages.Message):
    """Output only. BigQuery usage statistics over each of the predefined
    time ranges. Supported time ranges are `{"24H", "7D", "30D"}`.

    Messages:
      AdditionalProperty: An additional property for a
        UsageWithinTimeRangeValue object.

    Fields:
      additionalProperties: Additional properties of type
        UsageWithinTimeRangeValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a UsageWithinTimeRangeValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1UsageStats attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudDatacatalogV1UsageStats', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)