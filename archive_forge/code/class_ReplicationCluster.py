from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReplicationCluster(_messages.Message):
    """Primary-DR replica pair

  Fields:
    drReplica: Output only. read-only field that indicates if the replica is a
      dr_replica; not set for a primary.
    failoverDrReplicaName: Optional. If the instance is a primary instance,
      then this field identifies the disaster recovery (DR) replica. A DR
      replica is an optional configuration for Enterprise Plus edition
      instances. If the instance is a read replica, then the field is not set.
      Users can set this field to set a designated DR replica for a primary.
      Removing this field removes the DR replica.
    psaWriteEndpoint: Output only. If set, it indicates this instance has a
      private service access (PSA) dns endpoint that is pointing to the
      primary instance of the cluster. If this instance is the primary, the
      dns should be pointing to this instance. After Switchover or Replica
      failover, this DNS endpoint points to the promoted instance. This is a
      read-only field, returned to the user as information. This field can
      exist even if a standalone instance does not yet have a replica, or had
      a DR replica that was deleted.
  """
    drReplica = _messages.BooleanField(1)
    failoverDrReplicaName = _messages.StringField(2)
    psaWriteEndpoint = _messages.StringField(3)