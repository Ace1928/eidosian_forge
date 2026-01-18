from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PickTimeSeriesFilter(_messages.Message):
    """Describes a ranking-based time series filter. Each input time series is
  ranked with an aligner. The filter will allow up to num_time_series time
  series to pass through it, selecting them based on the relative ranking.For
  example, if ranking_method is METHOD_MEAN,direction is BOTTOM, and
  num_time_series is 3, then the 3 times series with the lowest mean values
  will pass through the filter.

  Enums:
    DirectionValueValuesEnum: How to use the ranking to select time series
      that pass through the filter.
    RankingMethodValueValuesEnum: ranking_method is applied to each time
      series independently to produce the value which will be used to compare
      the time series to other time series.

  Fields:
    direction: How to use the ranking to select time series that pass through
      the filter.
    interval: Select the top N streams/time series within this time interval
    numTimeSeries: How many time series to allow to pass through the filter.
    rankingMethod: ranking_method is applied to each time series independently
      to produce the value which will be used to compare the time series to
      other time series.
  """

    class DirectionValueValuesEnum(_messages.Enum):
        """How to use the ranking to select time series that pass through the
    filter.

    Values:
      DIRECTION_UNSPECIFIED: Not allowed. You must specify a different
        Direction if you specify a PickTimeSeriesFilter.
      TOP: Pass the highest num_time_series ranking inputs.
      BOTTOM: Pass the lowest num_time_series ranking inputs.
    """
        DIRECTION_UNSPECIFIED = 0
        TOP = 1
        BOTTOM = 2

    class RankingMethodValueValuesEnum(_messages.Enum):
        """ranking_method is applied to each time series independently to produce
    the value which will be used to compare the time series to other time
    series.

    Values:
      METHOD_UNSPECIFIED: Not allowed. You must specify a different Method if
        you specify a PickTimeSeriesFilter.
      METHOD_MEAN: Select the mean of all values.
      METHOD_MAX: Select the maximum value.
      METHOD_MIN: Select the minimum value.
      METHOD_SUM: Compute the sum of all values.
      METHOD_LATEST: Select the most recent value.
    """
        METHOD_UNSPECIFIED = 0
        METHOD_MEAN = 1
        METHOD_MAX = 2
        METHOD_MIN = 3
        METHOD_SUM = 4
        METHOD_LATEST = 5
    direction = _messages.EnumField('DirectionValueValuesEnum', 1)
    interval = _messages.MessageField('Interval', 2)
    numTimeSeries = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    rankingMethod = _messages.EnumField('RankingMethodValueValuesEnum', 4)