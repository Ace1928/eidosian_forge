from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Scorecard(_messages.Message):
    """A widget showing the latest value of a metric, and how this value
  relates to one or more thresholds.

  Fields:
    blankView: Will cause the Scorecard to show only the value, with no
      indicator to its value relative to its thresholds.
    gaugeView: Will cause the scorecard to show a gauge chart.
    sparkChartView: Will cause the scorecard to show a spark chart.
    thresholds: The thresholds used to determine the state of the scorecard
      given the time series' current value. For an actual value x, the
      scorecard is in a danger state if x is less than or equal to a danger
      threshold that triggers below, or greater than or equal to a danger
      threshold that triggers above. Similarly, if x is above/below a warning
      threshold that triggers above/below, then the scorecard is in a warning
      state - unless x also puts it in a danger state. (Danger trumps
      warning.)As an example, consider a scorecard with the following four
      thresholds: { value: 90, category: 'DANGER', trigger: 'ABOVE', }, {
      value: 70, category: 'WARNING', trigger: 'ABOVE', }, { value: 10,
      category: 'DANGER', trigger: 'BELOW', }, { value: 20, category:
      'WARNING', trigger: 'BELOW', } Then: values less than or equal to 10
      would put the scorecard in a DANGER state, values greater than 10 but
      less than or equal to 20 a WARNING state, values strictly between 20 and
      70 an OK state, values greater than or equal to 70 but less than 90 a
      WARNING state, and values greater than or equal to 90 a DANGER state.
    timeSeriesQuery: Required. Fields for querying time series data from the
      Stackdriver metrics API.
  """
    blankView = _messages.MessageField('Empty', 1)
    gaugeView = _messages.MessageField('GaugeView', 2)
    sparkChartView = _messages.MessageField('SparkChartView', 3)
    thresholds = _messages.MessageField('Threshold', 4, repeated=True)
    timeSeriesQuery = _messages.MessageField('TimeSeriesQuery', 5)