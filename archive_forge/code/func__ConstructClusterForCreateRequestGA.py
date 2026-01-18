from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterForCreateRequestGA(alloydb_messages, args):
    """Returns the cluster for GA create request based on args."""
    cluster = alloydb_messages.Cluster()
    cluster.network = args.network
    cluster.initialUser = alloydb_messages.UserPassword(password=args.password, user='postgres')
    kms_key = flags.GetAndValidateKmsKeyName(args)
    if kms_key:
        encryption_config = alloydb_messages.EncryptionConfig()
        encryption_config.kmsKeyName = kms_key
        cluster.encryptionConfig = encryption_config
    if args.disable_automated_backup or args.automated_backup_days_of_week:
        cluster.automatedBackupPolicy = _ConstructAutomatedBackupPolicy(alloydb_messages, args)
    if args.enable_continuous_backup is not None or args.continuous_backup_recovery_window_days or args.continuous_backup_encryption_key:
        cluster.continuousBackupConfig = _ConstructContinuousBackupConfig(alloydb_messages, args)
    if args.allocated_ip_range_name:
        cluster.networkConfig = alloydb_messages.NetworkConfig(network=args.network, allocatedIpRange=args.allocated_ip_range_name)
    if args.enable_private_service_connect:
        cluster.pscConfig = alloydb_messages.PscConfig(pscEnabled=True)
    cluster.databaseVersion = args.database_version
    return cluster