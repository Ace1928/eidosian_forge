from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaResult(_messages.Message):
    """(Auto-)arima fitting result. Wrap everything in ArimaResult for easier
  refactoring if we want to use model-specific iteration results.

  Enums:
    SeasonalPeriodsValueListEntryValuesEnum:

  Fields:
    arimaModelInfo: This message is repeated because there are multiple arima
      models fitted in auto-arima. For non-auto-arima model, its size is one.
    seasonalPeriods: Seasonal periods. Repeated because multiple periods are
      supported for one time series.
  """

    class SeasonalPeriodsValueListEntryValuesEnum(_messages.Enum):
        """SeasonalPeriodsValueListEntryValuesEnum enum type.

    Values:
      SEASONAL_PERIOD_TYPE_UNSPECIFIED: Unspecified seasonal period.
      NO_SEASONALITY: No seasonality
      DAILY: Daily period, 24 hours.
      WEEKLY: Weekly period, 7 days.
      MONTHLY: Monthly period, 30 days or irregular.
      QUARTERLY: Quarterly period, 90 days or irregular.
      YEARLY: Yearly period, 365 days or irregular.
    """
        SEASONAL_PERIOD_TYPE_UNSPECIFIED = 0
        NO_SEASONALITY = 1
        DAILY = 2
        WEEKLY = 3
        MONTHLY = 4
        QUARTERLY = 5
        YEARLY = 6
    arimaModelInfo = _messages.MessageField('ArimaModelInfo', 1, repeated=True)
    seasonalPeriods = _messages.EnumField('SeasonalPeriodsValueListEntryValuesEnum', 2, repeated=True)