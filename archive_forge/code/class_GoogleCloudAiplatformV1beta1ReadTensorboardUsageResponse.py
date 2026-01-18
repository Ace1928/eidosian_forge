from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadTensorboardUsageResponse(_messages.Message):
    """Response message for TensorboardService.ReadTensorboardUsage.

  Messages:
    MonthlyUsageDataValue: Maps year-month (YYYYMM) string to per month usage
      data.

  Fields:
    monthlyUsageData: Maps year-month (YYYYMM) string to per month usage data.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MonthlyUsageDataValue(_messages.Message):
        """Maps year-month (YYYYMM) string to per month usage data.

    Messages:
      AdditionalProperty: An additional property for a MonthlyUsageDataValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MonthlyUsageDataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MonthlyUsageDataValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1ReadTensorboardUsageResponsePerMo
          nthUsageData attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1ReadTensorboardUsageResponsePerMonthUsageData', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    monthlyUsageData = _messages.MessageField('MonthlyUsageDataValue', 1)