from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructuredQuery(_messages.Message):
    """A Firestore query. The query stages are executed in the following order:
  1. from 2. where 3. select 4. order_by + start_at + end_at 5. offset 6.
  limit

  Fields:
    endAt: A potential prefix of a position in the result set to end the query
      at. This is similar to `START_AT` but with it controlling the end
      position rather than the start position. Requires: * The number of
      values cannot be greater than the number of fields specified in the
      `ORDER BY` clause.
    findNearest: Optional. A potential Nearest Neighbors Search. Applies after
      all other filters and ordering. Finds the closest vector embeddings to
      the given query vector.
    from_: The collections to query.
    limit: The maximum number of results to return. Applies after all other
      constraints. Requires: * The value must be greater than or equal to zero
      if specified.
    offset: The number of documents to skip before returning the first result.
      This applies after the constraints specified by the `WHERE`, `START AT`,
      & `END AT` but before the `LIMIT` clause. Requires: * The value must be
      greater than or equal to zero if specified.
    orderBy: The order to apply to the query results. Firestore allows callers
      to provide a full ordering, a partial ordering, or no ordering at all.
      In all cases, Firestore guarantees a stable ordering through the
      following rules: * The `order_by` is required to reference all fields
      used with an inequality filter. * All fields that are required to be in
      the `order_by` but are not already present are appended in
      lexicographical ordering of the field name. * If an order on `__name__`
      is not specified, it is appended by default. Fields are appended with
      the same sort direction as the last order specified, or 'ASCENDING' if
      no order was specified. For example: * `ORDER BY a` becomes `ORDER BY a
      ASC, __name__ ASC` * `ORDER BY a DESC` becomes `ORDER BY a DESC,
      __name__ DESC` * `WHERE a > 1` becomes `WHERE a > 1 ORDER BY a ASC,
      __name__ ASC` * `WHERE __name__ > ... AND a > 1` becomes `WHERE __name__
      > ... AND a > 1 ORDER BY a ASC, __name__ ASC`
    select: Optional sub-set of the fields to return. This acts as a
      DocumentMask over the documents returned from a query. When not set,
      assumes that the caller wants all fields returned.
    startAt: A potential prefix of a position in the result set to start the
      query at. The ordering of the result set is based on the `ORDER BY`
      clause of the original query. ``` SELECT * FROM k WHERE a = 1 AND b > 2
      ORDER BY b ASC, __name__ ASC; ``` This query's results are ordered by
      `(b ASC, __name__ ASC)`. Cursors can reference either the full ordering
      or a prefix of the location, though it cannot reference more fields than
      what are in the provided `ORDER BY`. Continuing off the example above,
      attaching the following start cursors will have varying impact: - `START
      BEFORE (2, /k/123)`: start the query right before `a = 1 AND b > 2 AND
      __name__ > /k/123`. - `START AFTER (10)`: start the query right after `a
      = 1 AND b > 10`. Unlike `OFFSET` which requires scanning over the first
      N results to skip, a start cursor allows the query to begin at a logical
      position. This position is not required to match an actual result, it
      will scan forward from this position to find the next document.
      Requires: * The number of values cannot be greater than the number of
      fields specified in the `ORDER BY` clause.
    where: The filter to apply.
  """
    endAt = _messages.MessageField('Cursor', 1)
    findNearest = _messages.MessageField('FindNearest', 2)
    from_ = _messages.MessageField('CollectionSelector', 3, repeated=True)
    limit = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    offset = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    orderBy = _messages.MessageField('Order', 6, repeated=True)
    select = _messages.MessageField('Projection', 7)
    startAt = _messages.MessageField('Cursor', 8)
    where = _messages.MessageField('Filter', 9)