from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesQuery(_messages.Message):
    """TimeSeriesQuery collects the set of supported methods for querying time
  series data from the Stackdriver metrics API.

  Fields:
    opsAnalyticsQuery: Preview: A query used to fetch a time series, category
      series, or numeric series with SQL. This is a preview feature and may be
      subject to change before final release.
    outputFullDuration: Optional. If set, Cloud Monitoring will treat the full
      query duration as the alignment period so that there will be only 1
      output value.*Note: This could override the configured alignment period
      except for the cases where a series of data points are expected, like -
      XyChart - Scorecard's spark chart
    prometheusQuery: A query used to fetch time series with PromQL.
    timeSeriesFilter: Filter parameters to fetch time series.
    timeSeriesFilterRatio: Parameters to fetch a ratio between two time series
      filters.
    timeSeriesQueryLanguage: A query used to fetch time series with MQL.
    unitOverride: The unit of data contained in fetched time series. If non-
      empty, this unit will override any unit that accompanies fetched data.
      The format is the same as the unit (https://cloud.google.com/monitoring/
      api/ref_v3/rest/v3/projects.metricDescriptors) field in
      MetricDescriptor.
  """
    opsAnalyticsQuery = _messages.MessageField('OpsAnalyticsQuery', 1)
    outputFullDuration = _messages.BooleanField(2)
    prometheusQuery = _messages.StringField(3)
    timeSeriesFilter = _messages.MessageField('TimeSeriesFilter', 4)
    timeSeriesFilterRatio = _messages.MessageField('TimeSeriesFilterRatio', 5)
    timeSeriesQueryLanguage = _messages.StringField(6)
    unitOverride = _messages.StringField(7)