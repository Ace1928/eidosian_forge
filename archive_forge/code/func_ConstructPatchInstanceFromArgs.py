from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@classmethod
def ConstructPatchInstanceFromArgs(cls, sql_messages, args, original, instance_ref=None, release_track=DEFAULT_RELEASE_TRACK):
    """Constructs Instance for patch request from base instance and args."""
    instance_resource = cls._ConstructBaseInstanceFromArgs(sql_messages, args, original, instance_ref)
    instance_resource.databaseVersion = ParseDatabaseVersion(sql_messages, args.database_version)
    instance_resource.maintenanceVersion = args.maintenance_version
    instance_resource.settings = cls._ConstructPatchSettingsFromArgs(sql_messages, args, original, release_track)
    if args.upgrade_sql_network_architecture:
        instance_resource.sqlNetworkArchitecture = sql_messages.DatabaseInstance.SqlNetworkArchitectureValueValuesEnum.NEW_NETWORK_ARCHITECTURE
    if args.IsSpecified('simulate_maintenance_event'):
        instance_resource.maintenanceVersion = original.maintenanceVersion
        api_util.InstancesV1Beta4.PrintAndConfirmSimulatedMaintenanceEvent()
    if args.IsSpecified('maintenance_version') and args.maintenance_version == original.maintenanceVersion:
        api_util.InstancesV1Beta4.PrintAndConfirmSimulatedMaintenanceEvent()
    if IsBetaOrNewer(release_track):
        if args.IsKnownAndSpecified('failover_dr_replica_name'):
            replication_cluster = sql_messages.ReplicationCluster()
            replication_cluster.failoverDrReplicaName = args.failover_dr_replica_name
            instance_resource.replicationCluster = replication_cluster
        if args.IsKnownAndSpecified('clear_failover_dr_replica_name'):
            if instance_resource.replicationCluster is not None:
                instance_resource.replicationCluster.ClearFailoverDrReplicaName()
    return instance_resource