from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaForecastingMetrics(_messages.Message):
    """Model evaluation metrics for ARIMA forecasting models.

  Enums:
    SeasonalPeriodsValueListEntryValuesEnum:

  Fields:
    arimaFittingMetrics: Arima model fitting metrics.
    arimaSingleModelForecastingMetrics: Repeated as there can be many metric
      sets (one for each model) in auto-arima and the large-scale case.
    hasDrift: Whether Arima model fitted with drift or not. It is always false
      when d is not 1.
    nonSeasonalOrder: Non-seasonal order.
    seasonalPeriods: Seasonal periods. Repeated because multiple periods are
      supported for one time series.
    timeSeriesId: Id to differentiate different time series for the large-
      scale case.
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
    arimaFittingMetrics = _messages.MessageField('ArimaFittingMetrics', 1, repeated=True)
    arimaSingleModelForecastingMetrics = _messages.MessageField('ArimaSingleModelForecastingMetrics', 2, repeated=True)
    hasDrift = _messages.BooleanField(3, repeated=True)
    nonSeasonalOrder = _messages.MessageField('ArimaOrder', 4, repeated=True)
    seasonalPeriods = _messages.EnumField('SeasonalPeriodsValueListEntryValuesEnum', 5, repeated=True)
    timeSeriesId = _messages.StringField(6, repeated=True)