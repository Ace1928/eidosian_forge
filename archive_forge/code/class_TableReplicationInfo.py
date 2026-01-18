from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableReplicationInfo(_messages.Message):
    """Replication info of a table created using `AS REPLICA` DDL like: `CREATE
  MATERIALIZED VIEW mv1 AS REPLICA OF src_mv`

  Enums:
    ReplicationStatusValueValuesEnum: Optional. Output only. Replication
      status of configured replication.

  Fields:
    replicatedSourceLastRefreshTime: Optional. Output only. If source is a
      materialized view, this field signifies the last refresh time of the
      source.
    replicationError: Optional. Output only. Replication error that will
      permanently stopped table replication.
    replicationIntervalMs: Required. Specifies the interval at which the
      source table is polled for updates.
    replicationStatus: Optional. Output only. Replication status of configured
      replication.
    sourceTable: Required. Source table reference that is replicated.
  """

    class ReplicationStatusValueValuesEnum(_messages.Enum):
        """Optional. Output only. Replication status of configured replication.

    Values:
      REPLICATION_STATUS_UNSPECIFIED: Default value.
      ACTIVE: Replication is Active with no errors.
      SOURCE_DELETED: Source object is deleted.
      PERMISSION_DENIED: Source revoked replication permissions.
      UNSUPPORTED_CONFIGURATION: Source configuration doesn't allow
        replication.
    """
        REPLICATION_STATUS_UNSPECIFIED = 0
        ACTIVE = 1
        SOURCE_DELETED = 2
        PERMISSION_DENIED = 3
        UNSUPPORTED_CONFIGURATION = 4
    replicatedSourceLastRefreshTime = _messages.IntegerField(1)
    replicationError = _messages.MessageField('ErrorProto', 2)
    replicationIntervalMs = _messages.IntegerField(3)
    replicationStatus = _messages.EnumField('ReplicationStatusValueValuesEnum', 4)
    sourceTable = _messages.MessageField('TableReference', 5)