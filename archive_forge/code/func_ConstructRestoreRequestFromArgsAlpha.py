from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def ConstructRestoreRequestFromArgsAlpha(alloydb_messages, location_ref, resource_parser, args):
    """Returns the cluster restore request for Alpha track based on args."""
    cluster_resource = _ConstructClusterResourceForRestoreRequestAlpha(alloydb_messages, args)
    backup_source, continuous_backup_source = _ConstructBackupAndContinuousBackupSourceForRestoreRequest(alloydb_messages, resource_parser, args)
    return alloydb_messages.AlloydbProjectsLocationsClustersRestoreRequest(parent=location_ref.RelativeName(), restoreClusterRequest=alloydb_messages.RestoreClusterRequest(backupSource=backup_source, continuousBackupSource=continuous_backup_source, clusterId=args.cluster, cluster=cluster_resource))