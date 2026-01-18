from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecondaryAggregationCrossSeriesReducerValueValuesEnum(_messages.Enum):
    """The reduction operation to be used to combine time series into a
    single time series, where the value of each data point in the resulting
    series is a function of all the already aligned values in the input time
    series.Not all reducer operations can be applied to all time series. The
    valid choices depend on the metric_kind and the value_type of the original
    time series. Reduction can yield a time series with a different
    metric_kind or value_type than the input time series.Time series data must
    first be aligned (see per_series_aligner) in order to perform cross-time
    series reduction. If cross_series_reducer is specified, then
    per_series_aligner must be specified, and must not be ALIGN_NONE. An
    alignment_period must also be specified; otherwise, an error is returned.

    Values:
      REDUCE_NONE: No cross-time series reduction. The output of the Aligner
        is returned.
      REDUCE_MEAN: Reduce by computing the mean value across time series for
        each alignment period. This reducer is valid for DELTA and GAUGE
        metrics with numeric or distribution values. The value_type of the
        output is DOUBLE.
      REDUCE_MIN: Reduce by computing the minimum value across time series for
        each alignment period. This reducer is valid for DELTA and GAUGE
        metrics with numeric values. The value_type of the output is the same
        as the value_type of the input.
      REDUCE_MAX: Reduce by computing the maximum value across time series for
        each alignment period. This reducer is valid for DELTA and GAUGE
        metrics with numeric values. The value_type of the output is the same
        as the value_type of the input.
      REDUCE_SUM: Reduce by computing the sum across time series for each
        alignment period. This reducer is valid for DELTA and GAUGE metrics
        with numeric and distribution values. The value_type of the output is
        the same as the value_type of the input.
      REDUCE_STDDEV: Reduce by computing the standard deviation across time
        series for each alignment period. This reducer is valid for DELTA and
        GAUGE metrics with numeric or distribution values. The value_type of
        the output is DOUBLE.
      REDUCE_COUNT: Reduce by computing the number of data points across time
        series for each alignment period. This reducer is valid for DELTA and
        GAUGE metrics of numeric, Boolean, distribution, and string
        value_type. The value_type of the output is INT64.
      REDUCE_COUNT_TRUE: Reduce by computing the number of True-valued data
        points across time series for each alignment period. This reducer is
        valid for DELTA and GAUGE metrics of Boolean value_type. The
        value_type of the output is INT64.
      REDUCE_COUNT_FALSE: Reduce by computing the number of False-valued data
        points across time series for each alignment period. This reducer is
        valid for DELTA and GAUGE metrics of Boolean value_type. The
        value_type of the output is INT64.
      REDUCE_FRACTION_TRUE: Reduce by computing the ratio of the number of
        True-valued data points to the total number of data points for each
        alignment period. This reducer is valid for DELTA and GAUGE metrics of
        Boolean value_type. The output value is in the range 0.0, 1.0 and has
        value_type DOUBLE.
      REDUCE_PERCENTILE_99: Reduce by computing the 99th percentile
        (https://en.wikipedia.org/wiki/Percentile) of data points across time
        series for each alignment period. This reducer is valid for GAUGE and
        DELTA metrics of numeric and distribution type. The value of the
        output is DOUBLE.
      REDUCE_PERCENTILE_95: Reduce by computing the 95th percentile
        (https://en.wikipedia.org/wiki/Percentile) of data points across time
        series for each alignment period. This reducer is valid for GAUGE and
        DELTA metrics of numeric and distribution type. The value of the
        output is DOUBLE.
      REDUCE_PERCENTILE_50: Reduce by computing the 50th percentile
        (https://en.wikipedia.org/wiki/Percentile) of data points across time
        series for each alignment period. This reducer is valid for GAUGE and
        DELTA metrics of numeric and distribution type. The value of the
        output is DOUBLE.
      REDUCE_PERCENTILE_05: Reduce by computing the 5th percentile
        (https://en.wikipedia.org/wiki/Percentile) of data points across time
        series for each alignment period. This reducer is valid for GAUGE and
        DELTA metrics of numeric and distribution type. The value of the
        output is DOUBLE.
    """
    REDUCE_NONE = 0
    REDUCE_MEAN = 1
    REDUCE_MIN = 2
    REDUCE_MAX = 3
    REDUCE_SUM = 4
    REDUCE_STDDEV = 5
    REDUCE_COUNT = 6
    REDUCE_COUNT_TRUE = 7
    REDUCE_COUNT_FALSE = 8
    REDUCE_FRACTION_TRUE = 9
    REDUCE_PERCENTILE_99 = 10
    REDUCE_PERCENTILE_95 = 11
    REDUCE_PERCENTILE_50 = 12
    REDUCE_PERCENTILE_05 = 13