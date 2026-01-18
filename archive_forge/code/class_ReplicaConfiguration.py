from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReplicaConfiguration(_messages.Message):
    """Read-replica configuration for connecting to the primary instance.

  Fields:
    cascadableReplica: Optional. Specifies if a SQL Server replica is a
      cascadable replica. A cascadable replica is a SQL Server cross region
      replica that supports replica(s) under it.
    failoverTarget: Specifies if the replica is the failover target. If the
      field is set to `true` the replica will be designated as a failover
      replica. In case the primary instance fails, the replica instance will
      be promoted as the new primary instance. Only one replica can be
      specified as failover target, and the replica has to be in different
      zone with the primary instance.
    kind: This is always `sql#replicaConfiguration`.
    mysqlReplicaConfiguration: MySQL specific configuration when replicating
      from a MySQL on-premises primary instance. Replication configuration
      information such as the username, password, certificates, and keys are
      not stored in the instance metadata. The configuration information is
      used only to set up the replication connection and is stored by MySQL in
      a file named `master.info` in the data directory.
  """
    cascadableReplica = _messages.BooleanField(1)
    failoverTarget = _messages.BooleanField(2)
    kind = _messages.StringField(3)
    mysqlReplicaConfiguration = _messages.MessageField('MySqlReplicaConfiguration', 4)