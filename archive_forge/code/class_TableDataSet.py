from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableDataSet(_messages.Message):
    """Groups a time series query definition with table options.

  Fields:
    minAlignmentPeriod: Optional. The lower bound on data point frequency for
      this data set, implemented by specifying the minimum alignment period to
      use in a time series query For example, if the data is published once
      every 10 minutes, the min_alignment_period should be at least 10
      minutes. It would not make sense to fetch and align data at one minute
      intervals.
    tableDisplayOptions: Optional. Table display options for configuring how
      the table is rendered.
    tableTemplate: Optional. A template string for naming TimeSeries in the
      resulting data set. This should be a string with interpolations of the
      form ${label_name}, which will resolve to the label's value i.e.
      "${resource.labels.project_id}."
    timeSeriesQuery: Required. Fields for querying time series data from the
      Stackdriver metrics API.
  """
    minAlignmentPeriod = _messages.StringField(1)
    tableDisplayOptions = _messages.MessageField('TableDisplayOptions', 2)
    tableTemplate = _messages.StringField(3)
    timeSeriesQuery = _messages.MessageField('TimeSeriesQuery', 4)