from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseBackupConfig(self, volume, backup_config):
    """Parses Backup Config for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object.
      backup_config: the Backup Config message object.

    Returns:
      Volume message populated with Backup Config values.

    """
    backup_config_message = self.messages.BackupConfig()
    for backup_policy in backup_config.get('backup-policies', []):
        backup_config_message.backupPolicies.append(backup_policy)
    backup_config_message.backupVault = backup_config.get('backup-vault', '')
    backup_config_message.scheduledBackupEnabled = backup_config.get('enable-scheduled-backups', None)
    volume.backupConfig = backup_config_message