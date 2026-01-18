from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructuredAggregationQuery(_messages.Message):
    """Firestore query for running an aggregation over a StructuredQuery.

  Fields:
    aggregations: Optional. Series of aggregations to apply over the results
      of the `structured_query`. Requires: * A minimum of one and maximum of
      five aggregations per query.
    structuredQuery: Nested structured query.
  """
    aggregations = _messages.MessageField('Aggregation', 1, repeated=True)
    structuredQuery = _messages.MessageField('StructuredQuery', 2)