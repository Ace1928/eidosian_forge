from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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