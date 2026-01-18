from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsTimeSeriesListRequest(_messages.Message):
    """A MonitoringProjectsTimeSeriesListRequest object.

  Enums:
    AggregationCrossSeriesReducerValueValuesEnum: The reduction operation to
      be used to combine time series into a single time series, where the
      value of each data point in the resulting series is a function of all
      the already aligned values in the input time series.Not all reducer
      operations can be applied to all time series. The valid choices depend
      on the metric_kind and the value_type of the original time series.
      Reduction can yield a time series with a different metric_kind or
      value_type than the input time series.Time series data must first be
      aligned (see per_series_aligner) in order to perform cross-time series
      reduction. If cross_series_reducer is specified, then per_series_aligner
      must be specified, and must not be ALIGN_NONE. An alignment_period must
      also be specified; otherwise, an error is returned.
    AggregationPerSeriesAlignerValueValuesEnum: An Aligner describes how to
      bring the data points in a single time series into temporal alignment.
      Except for ALIGN_NONE, all alignments cause all the data points in an
      alignment_period to be mathematically grouped together, resulting in a
      single data point for each alignment_period with end timestamp at the
      end of the period.Not all alignment operations may be applied to all
      time series. The valid choices depend on the metric_kind and value_type
      of the original time series. Alignment can change the metric_kind or the
      value_type of the time series.Time series data must be aligned in order
      to perform cross-time series reduction. If cross_series_reducer is
      specified, then per_series_aligner must be specified and not equal to
      ALIGN_NONE and alignment_period must be specified; otherwise, an error
      is returned.
    SecondaryAggregationCrossSeriesReducerValueValuesEnum: The reduction
      operation to be used to combine time series into a single time series,
      where the value of each data point in the resulting series is a function
      of all the already aligned values in the input time series.Not all
      reducer operations can be applied to all time series. The valid choices
      depend on the metric_kind and the value_type of the original time
      series. Reduction can yield a time series with a different metric_kind
      or value_type than the input time series.Time series data must first be
      aligned (see per_series_aligner) in order to perform cross-time series
      reduction. If cross_series_reducer is specified, then per_series_aligner
      must be specified, and must not be ALIGN_NONE. An alignment_period must
      also be specified; otherwise, an error is returned.
    SecondaryAggregationPerSeriesAlignerValueValuesEnum: An Aligner describes
      how to bring the data points in a single time series into temporal
      alignment. Except for ALIGN_NONE, all alignments cause all the data
      points in an alignment_period to be mathematically grouped together,
      resulting in a single data point for each alignment_period with end
      timestamp at the end of the period.Not all alignment operations may be
      applied to all time series. The valid choices depend on the metric_kind
      and value_type of the original time series. Alignment can change the
      metric_kind or the value_type of the time series.Time series data must
      be aligned in order to perform cross-time series reduction. If
      cross_series_reducer is specified, then per_series_aligner must be
      specified and not equal to ALIGN_NONE and alignment_period must be
      specified; otherwise, an error is returned.
    ViewValueValuesEnum: Required. Specifies which information is returned
      about the time series.

  Fields:
    aggregation_alignmentPeriod: The alignment_period specifies a time
      interval, in seconds, that is used to divide the data in all the time
      series into consistent blocks of time. This will be done before the per-
      series aligner can be applied to the data.The value must be at least 60
      seconds. If a per-series aligner other than ALIGN_NONE is specified,
      this field is required or an error is returned. If no per-series aligner
      is specified, or the aligner ALIGN_NONE is specified, then this field is
      ignored.The maximum value of the alignment_period is 104 weeks (2 years)
      for charts, and 90,000 seconds (25 hours) for alerting policies.
    aggregation_crossSeriesReducer: The reduction operation to be used to
      combine time series into a single time series, where the value of each
      data point in the resulting series is a function of all the already
      aligned values in the input time series.Not all reducer operations can
      be applied to all time series. The valid choices depend on the
      metric_kind and the value_type of the original time series. Reduction
      can yield a time series with a different metric_kind or value_type than
      the input time series.Time series data must first be aligned (see
      per_series_aligner) in order to perform cross-time series reduction. If
      cross_series_reducer is specified, then per_series_aligner must be
      specified, and must not be ALIGN_NONE. An alignment_period must also be
      specified; otherwise, an error is returned.
    aggregation_groupByFields: The set of fields to preserve when
      cross_series_reducer is specified. The group_by_fields determine how the
      time series are partitioned into subsets prior to applying the
      aggregation operation. Each subset contains time series that have the
      same value for each of the grouping fields. Each individual time series
      is a member of exactly one subset. The cross_series_reducer is applied
      to each subset of time series. It is not possible to reduce across
      different resource types, so this field implicitly contains
      resource.type. Fields not specified in group_by_fields are aggregated
      away. If group_by_fields is not specified and all the time series have
      the same resource type, then the time series are aggregated into a
      single output time series. If cross_series_reducer is not defined, this
      field is ignored.
    aggregation_perSeriesAligner: An Aligner describes how to bring the data
      points in a single time series into temporal alignment. Except for
      ALIGN_NONE, all alignments cause all the data points in an
      alignment_period to be mathematically grouped together, resulting in a
      single data point for each alignment_period with end timestamp at the
      end of the period.Not all alignment operations may be applied to all
      time series. The valid choices depend on the metric_kind and value_type
      of the original time series. Alignment can change the metric_kind or the
      value_type of the time series.Time series data must be aligned in order
      to perform cross-time series reduction. If cross_series_reducer is
      specified, then per_series_aligner must be specified and not equal to
      ALIGN_NONE and alignment_period must be specified; otherwise, an error
      is returned.
    filter: Required. A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) that specifies
      which time series should be returned. The filter must specify a single
      metric type, and can additionally specify metric labels and other
      information. For example: metric.type =
      "compute.googleapis.com/instance/cpu/usage_time" AND
      metric.labels.instance_name = "my-instance-name"
    interval_endTime: Required. The end of the time interval.
    interval_startTime: Optional. The beginning of the time interval. The
      default value for the start time is the end time. The start time must
      not be later than the end time.
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name), organization
      or folder on which to execute the request. The format is:
      projects/[PROJECT_ID_OR_NUMBER] organizations/[ORGANIZATION_ID]
      folders/[FOLDER_ID]
    orderBy: Unsupported: must be left blank. The points in each time series
      are currently returned in reverse time order (most recent to oldest).
    pageSize: A positive number that is the maximum number of results to
      return. If page_size is empty or more than 100,000 results, the
      effective page_size is 100,000 results. If view is set to FULL, this is
      the maximum number of Points returned. If view is set to HEADERS, this
      is the maximum number of TimeSeries returned.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
    secondaryAggregation_alignmentPeriod: The alignment_period specifies a
      time interval, in seconds, that is used to divide the data in all the
      time series into consistent blocks of time. This will be done before the
      per-series aligner can be applied to the data.The value must be at least
      60 seconds. If a per-series aligner other than ALIGN_NONE is specified,
      this field is required or an error is returned. If no per-series aligner
      is specified, or the aligner ALIGN_NONE is specified, then this field is
      ignored.The maximum value of the alignment_period is 104 weeks (2 years)
      for charts, and 90,000 seconds (25 hours) for alerting policies.
    secondaryAggregation_crossSeriesReducer: The reduction operation to be
      used to combine time series into a single time series, where the value
      of each data point in the resulting series is a function of all the
      already aligned values in the input time series.Not all reducer
      operations can be applied to all time series. The valid choices depend
      on the metric_kind and the value_type of the original time series.
      Reduction can yield a time series with a different metric_kind or
      value_type than the input time series.Time series data must first be
      aligned (see per_series_aligner) in order to perform cross-time series
      reduction. If cross_series_reducer is specified, then per_series_aligner
      must be specified, and must not be ALIGN_NONE. An alignment_period must
      also be specified; otherwise, an error is returned.
    secondaryAggregation_groupByFields: The set of fields to preserve when
      cross_series_reducer is specified. The group_by_fields determine how the
      time series are partitioned into subsets prior to applying the
      aggregation operation. Each subset contains time series that have the
      same value for each of the grouping fields. Each individual time series
      is a member of exactly one subset. The cross_series_reducer is applied
      to each subset of time series. It is not possible to reduce across
      different resource types, so this field implicitly contains
      resource.type. Fields not specified in group_by_fields are aggregated
      away. If group_by_fields is not specified and all the time series have
      the same resource type, then the time series are aggregated into a
      single output time series. If cross_series_reducer is not defined, this
      field is ignored.
    secondaryAggregation_perSeriesAligner: An Aligner describes how to bring
      the data points in a single time series into temporal alignment. Except
      for ALIGN_NONE, all alignments cause all the data points in an
      alignment_period to be mathematically grouped together, resulting in a
      single data point for each alignment_period with end timestamp at the
      end of the period.Not all alignment operations may be applied to all
      time series. The valid choices depend on the metric_kind and value_type
      of the original time series. Alignment can change the metric_kind or the
      value_type of the time series.Time series data must be aligned in order
      to perform cross-time series reduction. If cross_series_reducer is
      specified, then per_series_aligner must be specified and not equal to
      ALIGN_NONE and alignment_period must be specified; otherwise, an error
      is returned.
    view: Required. Specifies which information is returned about the time
      series.
  """

    class AggregationCrossSeriesReducerValueValuesEnum(_messages.Enum):
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

    class AggregationPerSeriesAlignerValueValuesEnum(_messages.Enum):
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

    class SecondaryAggregationPerSeriesAlignerValueValuesEnum(_messages.Enum):
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

    class ViewValueValuesEnum(_messages.Enum):
        """Required. Specifies which information is returned about the time
    series.

    Values:
      FULL: Returns the identity of the metric(s), the time series, and the
        time series data.
      HEADERS: Returns the identity of the metric and the time series
        resource, but not the time series data.
    """
        FULL = 0
        HEADERS = 1
    aggregation_alignmentPeriod = _messages.StringField(1)
    aggregation_crossSeriesReducer = _messages.EnumField('AggregationCrossSeriesReducerValueValuesEnum', 2)
    aggregation_groupByFields = _messages.StringField(3, repeated=True)
    aggregation_perSeriesAligner = _messages.EnumField('AggregationPerSeriesAlignerValueValuesEnum', 4)
    filter = _messages.StringField(5)
    interval_endTime = _messages.StringField(6)
    interval_startTime = _messages.StringField(7)
    name = _messages.StringField(8, required=True)
    orderBy = _messages.StringField(9)
    pageSize = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(11)
    secondaryAggregation_alignmentPeriod = _messages.StringField(12)
    secondaryAggregation_crossSeriesReducer = _messages.EnumField('SecondaryAggregationCrossSeriesReducerValueValuesEnum', 13)
    secondaryAggregation_groupByFields = _messages.StringField(14, repeated=True)
    secondaryAggregation_perSeriesAligner = _messages.EnumField('SecondaryAggregationPerSeriesAlignerValueValuesEnum', 15)
    view = _messages.EnumField('ViewValueValuesEnum', 16)