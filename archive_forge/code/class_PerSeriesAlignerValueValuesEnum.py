from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerSeriesAlignerValueValuesEnum(_messages.Enum):
    """An Aligner describes how to bring the data points in a single time
    series into temporal alignment. Except for ALIGN_NONE, all alignments
    cause all the data points in an alignment_period to be mathematically
    grouped together, resulting in a single data point for each
    alignment_period with end timestamp at the end of the period.Not all
    alignment operations may be applied to all time series. The valid choices
    depend on the metric_kind and value_type of the original time series.
    Alignment can change the metric_kind or the value_type of the time
    series.Time series data must be aligned in order to perform cross-time
    series reduction. If cross_series_reducer is specified, then
    per_series_aligner must be specified and not equal to ALIGN_NONE and
    alignment_period must be specified; otherwise, an error is returned.

    Values:
      ALIGN_NONE: No alignment. Raw data is returned. Not valid if cross-
        series reduction is requested. The value_type of the result is the
        same as the value_type of the input.
      ALIGN_DELTA: Align and convert to DELTA. The output is delta = y1 -
        y0.This alignment is valid for CUMULATIVE and DELTA metrics. If the
        selected alignment period results in periods with no data, then the
        aligned value for such a period is created by interpolation. The
        value_type of the aligned result is the same as the value_type of the
        input.
      ALIGN_RATE: Align and convert to a rate. The result is computed as rate
        = (y1 - y0)/(t1 - t0), or "delta over time". Think of this aligner as
        providing the slope of the line that passes through the value at the
        start and at the end of the alignment_period.This aligner is valid for
        CUMULATIVE and DELTA metrics with numeric values. If the selected
        alignment period results in periods with no data, then the aligned
        value for such a period is created by interpolation. The output is a
        GAUGE metric with value_type DOUBLE.If, by "rate", you mean
        "percentage change", see the ALIGN_PERCENT_CHANGE aligner instead.
      ALIGN_INTERPOLATE: Align by interpolating between adjacent points around
        the alignment period boundary. This aligner is valid for GAUGE metrics
        with numeric values. The value_type of the aligned result is the same
        as the value_type of the input.
      ALIGN_NEXT_OLDER: Align by moving the most recent data point before the
        end of the alignment period to the boundary at the end of the
        alignment period. This aligner is valid for GAUGE metrics. The
        value_type of the aligned result is the same as the value_type of the
        input.
      ALIGN_MIN: Align the time series by returning the minimum value in each
        alignment period. This aligner is valid for GAUGE and DELTA metrics
        with numeric values. The value_type of the aligned result is the same
        as the value_type of the input.
      ALIGN_MAX: Align the time series by returning the maximum value in each
        alignment period. This aligner is valid for GAUGE and DELTA metrics
        with numeric values. The value_type of the aligned result is the same
        as the value_type of the input.
      ALIGN_MEAN: Align the time series by returning the mean value in each
        alignment period. This aligner is valid for GAUGE and DELTA metrics
        with numeric values. The value_type of the aligned result is DOUBLE.
      ALIGN_COUNT: Align the time series by returning the number of values in
        each alignment period. This aligner is valid for GAUGE and DELTA
        metrics with numeric or Boolean values. The value_type of the aligned
        result is INT64.
      ALIGN_SUM: Align the time series by returning the sum of the values in
        each alignment period. This aligner is valid for GAUGE and DELTA
        metrics with numeric and distribution values. The value_type of the
        aligned result is the same as the value_type of the input.
      ALIGN_STDDEV: Align the time series by returning the standard deviation
        of the values in each alignment period. This aligner is valid for
        GAUGE and DELTA metrics with numeric values. The value_type of the
        output is DOUBLE.
      ALIGN_COUNT_TRUE: Align the time series by returning the number of True
        values in each alignment period. This aligner is valid for GAUGE
        metrics with Boolean values. The value_type of the output is INT64.
      ALIGN_COUNT_FALSE: Align the time series by returning the number of
        False values in each alignment period. This aligner is valid for GAUGE
        metrics with Boolean values. The value_type of the output is INT64.
      ALIGN_FRACTION_TRUE: Align the time series by returning the ratio of the
        number of True values to the total number of values in each alignment
        period. This aligner is valid for GAUGE metrics with Boolean values.
        The output value is in the range 0.0, 1.0 and has value_type DOUBLE.
      ALIGN_PERCENTILE_99: Align the time series by using percentile
        aggregation (https://en.wikipedia.org/wiki/Percentile). The resulting
        data point in each alignment period is the 99th percentile of all data
        points in the period. This aligner is valid for GAUGE and DELTA
        metrics with distribution values. The output is a GAUGE metric with
        value_type DOUBLE.
      ALIGN_PERCENTILE_95: Align the time series by using percentile
        aggregation (https://en.wikipedia.org/wiki/Percentile). The resulting
        data point in each alignment period is the 95th percentile of all data
        points in the period. This aligner is valid for GAUGE and DELTA
        metrics with distribution values. The output is a GAUGE metric with
        value_type DOUBLE.
      ALIGN_PERCENTILE_50: Align the time series by using percentile
        aggregation (https://en.wikipedia.org/wiki/Percentile). The resulting
        data point in each alignment period is the 50th percentile of all data
        points in the period. This aligner is valid for GAUGE and DELTA
        metrics with distribution values. The output is a GAUGE metric with
        value_type DOUBLE.
      ALIGN_PERCENTILE_05: Align the time series by using percentile
        aggregation (https://en.wikipedia.org/wiki/Percentile). The resulting
        data point in each alignment period is the 5th percentile of all data
        points in the period. This aligner is valid for GAUGE and DELTA
        metrics with distribution values. The output is a GAUGE metric with
        value_type DOUBLE.
      ALIGN_PERCENT_CHANGE: Align and convert to a percentage change. This
        aligner is valid for GAUGE and DELTA metrics with numeric values. This
        alignment returns ((current - previous)/previous) * 100, where the
        value of previous is determined based on the alignment_period.If the
        values of current and previous are both 0, then the returned value is
        0. If only previous is 0, the returned value is infinity.A 10-minute
        moving mean is computed at each point of the alignment period prior to
        the above calculation to smooth the metric and prevent false positives
        from very short-lived spikes. The moving mean is only applicable for
        data whose values are >= 0. Any values < 0 are treated as a missing
        datapoint, and are ignored. While DELTA metrics are accepted by this
        alignment, special care should be taken that the values for the metric
        will always be positive. The output is a GAUGE metric with value_type
        DOUBLE.
    """
    ALIGN_NONE = 0
    ALIGN_DELTA = 1
    ALIGN_RATE = 2
    ALIGN_INTERPOLATE = 3
    ALIGN_NEXT_OLDER = 4
    ALIGN_MIN = 5
    ALIGN_MAX = 6
    ALIGN_MEAN = 7
    ALIGN_COUNT = 8
    ALIGN_SUM = 9
    ALIGN_STDDEV = 10
    ALIGN_COUNT_TRUE = 11
    ALIGN_COUNT_FALSE = 12
    ALIGN_FRACTION_TRUE = 13
    ALIGN_PERCENTILE_99 = 14
    ALIGN_PERCENTILE_95 = 15
    ALIGN_PERCENTILE_50 = 16
    ALIGN_PERCENTILE_05 = 17
    ALIGN_PERCENT_CHANGE = 18