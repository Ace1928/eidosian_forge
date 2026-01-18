from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionQueryResponse(_messages.Message):
    """The response for Firestore.PartitionQuery.

  Fields:
    nextPageToken: A page token that may be used to request an additional set
      of results, up to the number specified by `partition_count` in the
      PartitionQuery request. If blank, there are no more results.
    partitions: Partition results. Each partition is a split point that can be
      used by RunQuery as a starting or end point for the query results. The
      RunQuery requests must be made with the same query supplied to this
      PartitionQuery request. The partition cursors will be ordered according
      to same ordering as the results of the query supplied to PartitionQuery.
      For example, if a PartitionQuery request returns partition cursors A and
      B, running the following three queries will return the entire result set
      of the original query: * query, end_at A * query, start_at A, end_at B *
      query, start_at B An empty result may indicate that the query has too
      few results to be partitioned, or that the query is not yet supported
      for partitioning.
  """
    nextPageToken = _messages.StringField(1)
    partitions = _messages.MessageField('Cursor', 2, repeated=True)