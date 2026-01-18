from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartingBreakdown(_messages.Message):
    """Columns within the output of the previous step to use to break down the
  measures. We will generate one output measure for each value in the cross
  product of measure_columns plus the top limit values in each of the
  breakdown columns.In other words, if there is one measure column "foo" and
  two breakdown columns "bar" with values ("bar1","bar2") and "baz" with
  values ("baz1", "baz2"), we will end up with four output measures with
  names: foo_bar1_baz1, foo_bar1_baz2, foo_bar2_baz1, foo_bar2_baz2 Each of
  these measures will contain a subset of the values in "foo".

  Enums:
    SortOrderValueValuesEnum: Optional. The ordering that defines the behavior
      of limit. If limit is not zero, this may not be set to
      SORT_ORDER_NONE.Note that this will not control the ordering of the rows
      in the result table in any useful way. Use the top-level sort ordering
      for that purpose.

  Fields:
    column: Required. The name of the column containing the breakdown values.
    limit: Optional. Values to choose how many breakdowns to create for each
      measure. If limit is zero, all possible breakdowns will be generated. If
      not, limit determines how many breakdowns, and sort_aggregation
      determines the function we will use to sort the breakdowns.For example,
      if limit is 3, we will generate at most three breakdowns per measure. If
      sort_aggregation is "average" and sort_order is DESCENDING, those three
      will be chosen as the ones where the average of all the points in the
      breakdown set is the greatest.
    sortAggregation: Optional. The aggregation to apply to the measure values
      when choosing which breakdowns to generate. If sort_order is
      SORT_ORDER_NONE, this is not used.
    sortOrder: Optional. The ordering that defines the behavior of limit. If
      limit is not zero, this may not be set to SORT_ORDER_NONE.Note that this
      will not control the ordering of the rows in the result table in any
      useful way. Use the top-level sort ordering for that purpose.
  """

    class SortOrderValueValuesEnum(_messages.Enum):
        """Optional. The ordering that defines the behavior of limit. If limit is
    not zero, this may not be set to SORT_ORDER_NONE.Note that this will not
    control the ordering of the rows in the result table in any useful way.
    Use the top-level sort ordering for that purpose.

    Values:
      SORT_ORDER_UNSPECIFIED: Invalid value, do not use.
      SORT_ORDER_NONE: No sorting will be applied.
      SORT_ORDER_ASCENDING: The lowest-valued entries will be selected.
      SORT_ORDER_DESCENDING: The highest-valued entries will be selected.
    """
        SORT_ORDER_UNSPECIFIED = 0
        SORT_ORDER_NONE = 1
        SORT_ORDER_ASCENDING = 2
        SORT_ORDER_DESCENDING = 3
    column = _messages.StringField(1)
    limit = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    sortAggregation = _messages.MessageField('QueryStepAggregation', 3)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 4)