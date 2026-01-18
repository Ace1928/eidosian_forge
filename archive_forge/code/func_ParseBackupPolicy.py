from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseBackupPolicy(self, name=None, enabled=None, daily_backup_limit=None, weekly_backup_limit=None, monthly_backup_limit=None, description=None, labels=None):
    """Parses the command line arguments for Create Backup Policy into a message.

    Args:
      name: the name of the Backup Policy
      enabled: the Boolean value indicating whether or not backups are made
        automatically according to schedule.
      daily_backup_limit: the number of daily backups to keep.
      weekly_backup_limit: the number of weekly backups to keep.
      monthly_backup_limit: the number of monthly backups to keep.
      description: the description of the Backup Policy.
      labels: the parsed labels value

    Returns:
      The configuration that will be used as the request body for creating a
      Cloud NetApp Backup Policy.
    """
    backup_policy = self.messages.BackupPolicy()
    backup_policy.name = name
    backup_policy.enabled = enabled
    backup_policy.dailyBackupLimit = daily_backup_limit
    backup_policy.weeklyBackupLimit = weekly_backup_limit
    backup_policy.monthlyBackupLimit = monthly_backup_limit
    backup_policy.description = description
    backup_policy.labels = labels
    return backup_policy