from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructContinuousBackupConfig(alloydb_messages, args, update=False):
    """Returns the continuous backup config based on args."""
    continuous_backup_config = alloydb_messages.ContinuousBackupConfig()
    flags.ValidateContinuousBackupFlags(args, update)
    if args.enable_continuous_backup:
        continuous_backup_config.enabled = True
    elif args.enable_continuous_backup is False:
        continuous_backup_config.enabled = False
        return continuous_backup_config
    if args.continuous_backup_recovery_window_days:
        continuous_backup_config.recoveryWindowDays = args.continuous_backup_recovery_window_days
    kms_key = flags.GetAndValidateKmsKeyName(args, flag_overrides=flags.GetContinuousBackupKmsFlagOverrides())
    if kms_key:
        encryption_config = alloydb_messages.EncryptionConfig()
        encryption_config.kmsKeyName = kms_key
        continuous_backup_config.encryptionConfig = encryption_config
    return continuous_backup_config