from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesTable(_messages.Message):
    """A table that displays time series data.

  Enums:
    MetricVisualizationValueValuesEnum: Optional. Store rendering strategy

  Fields:
    columnSettings: Optional. The list of the persistent column settings for
      the table.
    dataSets: Required. The data displayed in this table.
    metricVisualization: Optional. Store rendering strategy
  """

    class MetricVisualizationValueValuesEnum(_messages.Enum):
        """Optional. Store rendering strategy

    Values:
      METRIC_VISUALIZATION_UNSPECIFIED: Unspecified state
      NUMBER: Default text rendering
      BAR: Horizontal bar rendering
    """
        METRIC_VISUALIZATION_UNSPECIFIED = 0
        NUMBER = 1
        BAR = 2
    columnSettings = _messages.MessageField('ColumnSettings', 1, repeated=True)
    dataSets = _messages.MessageField('TableDataSet', 2, repeated=True)
    metricVisualization = _messages.EnumField('MetricVisualizationValueValuesEnum', 3)