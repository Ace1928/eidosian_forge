from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatisticalTimeSeriesFilter(_messages.Message):
    """A filter that ranks streams based on their statistical relation to other
  streams in a request. Note: This field is deprecated and completely ignored
  by the API.

  Enums:
    RankingMethodValueValuesEnum: rankingMethod is applied to a set of time
      series, and then the produced value for each individual time series is
      used to compare a given time series to others. These are methods that
      cannot be applied stream-by-stream, but rather require the full context
      of a request to evaluate time series.

  Fields:
    numTimeSeries: How many time series to output.
    rankingMethod: rankingMethod is applied to a set of time series, and then
      the produced value for each individual time series is used to compare a
      given time series to others. These are methods that cannot be applied
      stream-by-stream, but rather require the full context of a request to
      evaluate time series.
  """

    class RankingMethodValueValuesEnum(_messages.Enum):
        """rankingMethod is applied to a set of time series, and then the
    produced value for each individual time series is used to compare a given
    time series to others. These are methods that cannot be applied stream-by-
    stream, but rather require the full context of a request to evaluate time
    series.

    Values:
      METHOD_UNSPECIFIED: Not allowed in well-formed requests.
      METHOD_CLUSTER_OUTLIER: Compute the outlier score of each stream.
    """
        METHOD_UNSPECIFIED = 0
        METHOD_CLUSTER_OUTLIER = 1
    numTimeSeries = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    rankingMethod = _messages.EnumField('RankingMethodValueValuesEnum', 2)