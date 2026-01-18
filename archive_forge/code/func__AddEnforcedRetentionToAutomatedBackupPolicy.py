from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _AddEnforcedRetentionToAutomatedBackupPolicy(backup_policy, args):
    if args.automated_backup_enforced_retention is not None:
        backup_policy.enforcedRetention = args.automated_backup_enforced_retention
    return backup_policy