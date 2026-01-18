from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoreResultsValueValuesEnum(_messages.Enum):
    """The state of the query after the current batch.

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