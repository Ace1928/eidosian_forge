from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class XyChart(_messages.Message):
    """A chart that displays data on a 2D (X and Y axes) plane.

  Fields:
    chartOptions: Display options for the chart.
    dataSets: Required. The data displayed in this chart.
    thresholds: Threshold lines drawn horizontally across the chart.
    timeshiftDuration: The duration used to display a comparison chart. A
      comparison chart simultaneously shows values from two similar-length
      time periods (e.g., week-over-week metrics). The duration must be
      positive, and it can only be applied to charts with data sets of LINE
      plot type.
    xAxis: The properties applied to the x-axis.
    y2Axis: The properties applied to the y2-axis.
    yAxis: The properties applied to the y-axis.
  """
    chartOptions = _messages.MessageField('ChartOptions', 1)
    dataSets = _messages.MessageField('DataSet', 2, repeated=True)
    thresholds = _messages.MessageField('Threshold', 3, repeated=True)
    timeshiftDuration = _messages.StringField(4)
    xAxis = _messages.MessageField('Axis', 5)
    y2Axis = _messages.MessageField('Axis', 6)
    yAxis = _messages.MessageField('Axis', 7)