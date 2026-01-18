from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationResultBatch(_messages.Message):
    """A batch of aggregation results produced by an aggregation query.

  Enums:
    MoreResultsValueValuesEnum: The state of the query after the current
      batch. Only COUNT(*) aggregations are supported in the initial launch.
      Therefore, expected result type is limited to `NO_MORE_RESULTS`.

  Fields:
    aggregationResults: The aggregation results for this batch.
    moreResults: The state of the query after the current batch. Only COUNT(*)
      aggregations are supported in the initial launch. Therefore, expected
      result type is limited to `NO_MORE_RESULTS`.
    readTime: Read timestamp this batch was returned from. In a single
      transaction, subsequent query result batches for the same query can have
      a greater timestamp. Each batch's read timestamp is valid for all
      preceding batches.
  """

    class MoreResultsValueValuesEnum(_messages.Enum):
        """The state of the query after the current batch. Only COUNT(*)
    aggregations are supported in the initial launch. Therefore, expected
    result type is limited to `NO_MORE_RESULTS`.

    Values:
      MORE_RESULTS_TYPE_UNSPECIFIED: Unspecified. This value is never used.
      NOT_FINISHED: There may be additional batches to fetch from this query.
      MORE_RESULTS_AFTER_LIMIT: The query is finished, but there may be more
        results after the limit.
      MORE_RESULTS_AFTER_CURSOR: The query is finished, but there may be more
        results after the end cursor.
      NO_MORE_RESULTS: The query is finished, and there are no more results.
    """
        MORE_RESULTS_TYPE_UNSPECIFIED = 0
        NOT_FINISHED = 1
        MORE_RESULTS_AFTER_LIMIT = 2
        MORE_RESULTS_AFTER_CURSOR = 3
        NO_MORE_RESULTS = 4
    aggregationResults = _messages.MessageField('AggregationResult', 1, repeated=True)
    moreResults = _messages.EnumField('MoreResultsValueValuesEnum', 2)
    readTime = _messages.StringField(3)