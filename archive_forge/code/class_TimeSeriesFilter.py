from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesFilter(_messages.Message):
    """A filter that defines a subset of time series data that is displayed in
  a widget. Time series data is fetched using the ListTimeSeries (https://clou
  d.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list) method.

  Fields:
    aggregation: By default, the raw time series data is returned. Use this
      field to combine multiple time series for different views of the data.
    filter: Required. The monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) that identifies the
      metric types, resources, and projects to query.
    pickTimeSeriesFilter: Ranking based time series filter.
    secondaryAggregation: Apply a second aggregation after aggregation is
      applied.
    statisticalTimeSeriesFilter: Statistics based time series filter. Note:
      This field is deprecated and completely ignored by the API.
  """
    aggregation = _messages.MessageField('Aggregation', 1)
    filter = _messages.StringField(2)
    pickTimeSeriesFilter = _messages.MessageField('PickTimeSeriesFilter', 3)
    secondaryAggregation = _messages.MessageField('Aggregation', 4)
    statisticalTimeSeriesFilter = _messages.MessageField('StatisticalTimeSeriesFilter', 5)