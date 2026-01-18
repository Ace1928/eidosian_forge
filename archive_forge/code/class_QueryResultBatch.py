from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryResultBatch(_messages.Message):
    """A batch of results produced by a query.

  Enums:
    EntityResultTypeValueValuesEnum: The result type for every entity in
      `entity_results`.
    MoreResultsValueValuesEnum: The state of the query after the current
      batch.

  Fields:
    endCursor: A cursor that points to the position after the last result in
      the batch.
    entityResultType: The result type for every entity in `entity_results`.
    entityResults: The results for this batch.
    moreResults: The state of the query after the current batch.
    readTime: Read timestamp this batch was returned from. This applies to the
      range of results from the query's `start_cursor` (or the beginning of
      the query if no cursor was given) to this batch's `end_cursor` (not the
      query's `end_cursor`). In a single transaction, subsequent query result
      batches for the same query can have a greater timestamp. Each batch's
      read timestamp is valid for all preceding batches. This value will not
      be set for eventually consistent queries in Cloud Datastore.
    skippedCursor: A cursor that points to the position after the last skipped
      result. Will be set when `skipped_results` != 0.
    skippedResults: The number of results skipped, typically because of an
      offset.
    snapshotVersion: The version number of the snapshot this batch was
      returned from. This applies to the range of results from the query's
      `start_cursor` (or the beginning of the query if no cursor was given) to
      this batch's `end_cursor` (not the query's `end_cursor`). In a single
      transaction, subsequent query result batches for the same query can have
      a greater snapshot version number. Each batch's snapshot version is
      valid for all preceding batches. The value will be zero for eventually
      consistent queries.
  """

    class EntityResultTypeValueValuesEnum(_messages.Enum):
        """The result type for every entity in `entity_results`.

    Values:
      RESULT_TYPE_UNSPECIFIED: Unspecified. This value is never used.
      FULL: The key and properties.
      PROJECTION: A projected subset of properties. The entity may have no
        key.
      KEY_ONLY: Only the key.
    """
        RESULT_TYPE_UNSPECIFIED = 0
        FULL = 1
        PROJECTION = 2
        KEY_ONLY = 3

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
    endCursor = _messages.BytesField(1)
    entityResultType = _messages.EnumField('EntityResultTypeValueValuesEnum', 2)
    entityResults = _messages.MessageField('EntityResult', 3, repeated=True)
    moreResults = _messages.EnumField('MoreResultsValueValuesEnum', 4)
    readTime = _messages.StringField(5)
    skippedCursor = _messages.BytesField(6)
    skippedResults = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    snapshotVersion = _messages.IntegerField(8)