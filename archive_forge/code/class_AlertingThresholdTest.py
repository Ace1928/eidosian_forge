from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertingThresholdTest(_messages.Message):
    """A test that compares some LHS against a threshold. NOTE: Only
  RowCountThreshold is currently supported.

  Enums:
    ComparisonValueValuesEnum: Required. The comparison to be applied in the
      __alert_result condition.

  Fields:
    aggregateValueThreshold: A value threshold comparison that includes an
      aggregation of the value column.
    comparison: Required. The comparison to be applied in the __alert_result
      condition.
    rowCountThreshold: A threshold based on the number of rows present.
    threshold: Required. The threshold that will be used as the RHS of a
      comparison.
    valueThreshold: A value threshold comparison.
  """

    class ComparisonValueValuesEnum(_messages.Enum):
        """Required. The comparison to be applied in the __alert_result
    condition.

    Values:
      COMPARISON_TYPE_UNSPECIFIED: No comparison relationship is specified.
      COMPARISON_GT: True if the aggregate / value_column is greater than the
        threshold.
      COMPARISON_GE: True if the aggregate / value_column is greater than or
        equal to the threshold.
      COMPARISON_LT: True if the aggregate / value_column is less than the
        threshold.
      COMPARISON_LE: True if the aggregate / value_column is less than or
        equal to the threshold.
      COMPARISON_EQ: True if the aggregate / value_column is equal to the
        threshold.
      COMPARISON_NE: True if the aggregate / value_column is not equal to the
        threshold.
    """
        COMPARISON_TYPE_UNSPECIFIED = 0
        COMPARISON_GT = 1
        COMPARISON_GE = 2
        COMPARISON_LT = 3
        COMPARISON_LE = 4
        COMPARISON_EQ = 5
        COMPARISON_NE = 6
    aggregateValueThreshold = _messages.MessageField('AggregateValueThreshold', 1)
    comparison = _messages.EnumField('ComparisonValueValuesEnum', 2)
    rowCountThreshold = _messages.MessageField('RowCountThreshold', 3)
    threshold = _messages.FloatField(4)
    valueThreshold = _messages.MessageField('ValueThreshold', 5)