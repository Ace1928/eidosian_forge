from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Breakdown(_messages.Message):
    """Preview: A breakdown is an aggregation applied to the measures over a
  specified column. A breakdown can result in multiple series across a
  category for the provided measure. This is a preview feature and may be
  subject to change before final release.

  Enums:
    SortOrderValueValuesEnum: Required. The sort order is applied to the
      values of the breakdown column.

  Fields:
    aggregationFunction: Required. The Aggregation function is applied across
      all data in each breakdown created.
    column: Required. The name of the column in the dataset containing the
      breakdown values.
    limit: Required. A limit to the number of breakdowns. If set to zero then
      all possible breakdowns are applied. The list of breakdowns is dependent
      on the value of the sort_order field.
    sortOrder: Required. The sort order is applied to the values of the
      breakdown column.
  """

    class SortOrderValueValuesEnum(_messages.Enum):
        """Required. The sort order is applied to the values of the breakdown
    column.

    Values:
      SORT_ORDER_UNSPECIFIED: An unspecified sort order. This option is
        invalid when sorting is required.
      SORT_ORDER_NONE: No sorting is applied.
      SORT_ORDER_ASCENDING: The lowest-valued entries are selected first.
      SORT_ORDER_DESCENDING: The highest-valued entries are selected first.
    """
        SORT_ORDER_UNSPECIFIED = 0
        SORT_ORDER_NONE = 1
        SORT_ORDER_ASCENDING = 2
        SORT_ORDER_DESCENDING = 3
    aggregationFunction = _messages.MessageField('AggregationFunction', 1)
    column = _messages.StringField(2)
    limit = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 4)