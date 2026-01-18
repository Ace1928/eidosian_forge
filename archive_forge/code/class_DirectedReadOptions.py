from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectedReadOptions(_messages.Message):
    """The DirectedReadOptions can be used to indicate which replicas or
  regions should be used for non-transactional reads or queries.
  DirectedReadOptions may only be specified for a read-only transaction,
  otherwise the API will return an `INVALID_ARGUMENT` error.

  Fields:
    excludeReplicas: Exclude_replicas indicates that specified replicas should
      be excluded from serving requests. Spanner will not route requests to
      the replicas in this list.
    includeReplicas: Include_replicas indicates the order of replicas (as they
      appear in this list) to process the request. If auto_failover_disabled
      is set to true and all replicas are exhausted without finding a healthy
      replica, Spanner will wait for a replica in the list to become
      available, requests may fail due to `DEADLINE_EXCEEDED` errors.
  """
    excludeReplicas = _messages.MessageField('ExcludeReplicas', 1)
    includeReplicas = _messages.MessageField('IncludeReplicas', 2)