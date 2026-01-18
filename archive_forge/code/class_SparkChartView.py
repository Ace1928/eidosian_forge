from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkChartView(_messages.Message):
    """A sparkChart is a small chart suitable for inclusion in a table-cell or
  inline in text. This message contains the configuration for a sparkChart to
  show up on a Scorecard, showing recent trends of the scorecard's timeseries.

  Enums:
    SparkChartTypeValueValuesEnum: Required. The type of sparkchart to show in
      this chartView.

  Fields:
    minAlignmentPeriod: The lower bound on data point frequency in the chart
      implemented by specifying the minimum alignment period to use in a time
      series query. For example, if the data is published once every 10
      minutes it would not make sense to fetch and align data at one minute
      intervals. This field is optional and exists only as a hint.
    sparkChartType: Required. The type of sparkchart to show in this
      chartView.
  """

    class SparkChartTypeValueValuesEnum(_messages.Enum):
        """Required. The type of sparkchart to show in this chartView.

    Values:
      SPARK_CHART_TYPE_UNSPECIFIED: Not allowed in well-formed requests.
      SPARK_LINE: The sparkline will be rendered as a small line chart.
      SPARK_BAR: The sparkbar will be rendered as a small bar chart.
    """
        SPARK_CHART_TYPE_UNSPECIFIED = 0
        SPARK_LINE = 1
        SPARK_BAR = 2
    minAlignmentPeriod = _messages.StringField(1)
    sparkChartType = _messages.EnumField('SparkChartTypeValueValuesEnum', 2)