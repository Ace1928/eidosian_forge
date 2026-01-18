from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1UsageSignal(_messages.Message):
    """The set of all usage signals that we store in Data Catalog.

  Messages:
    UsageWithinTimeRangeValue: Usage statistics over each of the pre-defined
      time ranges, supported strings for time ranges are {"24H", "7D", "30D"}.

  Fields:
    updateTime: The timestamp of the end of the usage statistics duration.
    usageWithinTimeRange: Usage statistics over each of the pre-defined time
      ranges, supported strings for time ranges are {"24H", "7D", "30D"}.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UsageWithinTimeRangeValue(_messages.Message):
        """Usage statistics over each of the pre-defined time ranges, supported
    strings for time ranges are {"24H", "7D", "30D"}.

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
        value: A GoogleCloudDatacatalogV1beta1UsageStats attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1beta1UsageStats', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    updateTime = _messages.StringField(1)
    usageWithinTimeRange = _messages.MessageField('UsageWithinTimeRangeValue', 2)