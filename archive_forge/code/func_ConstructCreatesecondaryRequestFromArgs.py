from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def ConstructCreatesecondaryRequestFromArgs(alloydb_messages, cluster_ref, args):
    """Returns the cluster create-secondary request based on args."""
    cluster = alloydb_messages.Cluster()
    cluster.secondaryConfig = alloydb_messages.SecondaryConfig(primaryClusterName=args.primary_cluster)
    kms_key = flags.GetAndValidateKmsKeyName(args)
    if kms_key:
        encryption_config = alloydb_messages.EncryptionConfig()
        encryption_config.kmsKeyName = kms_key
        cluster.encryptionConfig = encryption_config
    if args.enable_continuous_backup is not None or args.continuous_backup_recovery_window_days or args.continuous_backup_encryption_key:
        cluster.continuousBackupConfig = _ConstructContinuousBackupConfig(alloydb_messages, args)
    if args.allocated_ip_range_name:
        cluster.networkConfig = alloydb_messages.NetworkConfig(allocatedIpRange=args.allocated_ip_range_name)
    return alloydb_messages.AlloydbProjectsLocationsClustersCreatesecondaryRequest(cluster=cluster, clusterId=args.cluster, parent=cluster_ref.RelativeName())