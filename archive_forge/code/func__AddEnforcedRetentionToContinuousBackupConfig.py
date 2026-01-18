from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _AddEnforcedRetentionToContinuousBackupConfig(continuous_backup_config, args):
    if args.continuous_backup_enforced_retention is not None:
        continuous_backup_config.enforcedRetention = args.continuous_backup_enforced_retention
    return continuous_backup_config