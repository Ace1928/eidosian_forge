from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaSingleModelForecastingMetrics(_messages.Message):
    """Model evaluation metrics for a single ARIMA forecasting model.

  Enums:
    SeasonalPeriodsValueListEntryValuesEnum:

  Fields:
    arimaFittingMetrics: Arima fitting metrics.
    hasDrift: Is arima model fitted with drift or not. It is always false when
      d is not 1.
    hasHolidayEffect: If true, holiday_effect is a part of time series
      decomposition result.
    hasSpikesAndDips: If true, spikes_and_dips is a part of time series
      decomposition result.
    hasStepChanges: If true, step_changes is a part of time series
      decomposition result.
    nonSeasonalOrder: Non-seasonal order.
    seasonalPeriods: Seasonal periods. Repeated because multiple periods are
      supported for one time series.
    timeSeriesId: The time_series_id value for this time series. It will be
      one of the unique values from the time_series_id_column specified during
      ARIMA model training. Only present when time_series_id_column training
      option was used.
    timeSeriesIds: The tuple of time_series_ids identifying this time series.
      It will be one of the unique tuples of values present in the
      time_series_id_columns specified during ARIMA model training. Only
      present when time_series_id_columns training option was used and the
      order of values here are same as the order of time_series_id_columns.
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
    arimaFittingMetrics = _messages.MessageField('ArimaFittingMetrics', 1)
    hasDrift = _messages.BooleanField(2)
    hasHolidayEffect = _messages.BooleanField(3)
    hasSpikesAndDips = _messages.BooleanField(4)
    hasStepChanges = _messages.BooleanField(5)
    nonSeasonalOrder = _messages.MessageField('ArimaOrder', 6)
    seasonalPeriods = _messages.EnumField('SeasonalPeriodsValueListEntryValuesEnum', 7, repeated=True)
    timeSeriesId = _messages.StringField(8)
    timeSeriesIds = _messages.StringField(9, repeated=True)