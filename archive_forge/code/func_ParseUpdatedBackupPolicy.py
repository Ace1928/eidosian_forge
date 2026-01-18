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
def ParseUpdatedBackupPolicy(self, backup_policy, daily_backup_limit=None, weekly_backup_limit=None, monthly_backup_limit=None, enabled=None, description=None, labels=None):
    """Parses updates into a new Backup Policy."""
    if enabled is not None:
        backup_policy.enabled = enabled
    if daily_backup_limit is not None:
        backup_policy.dailyBackupLimit = daily_backup_limit
    if weekly_backup_limit is not None:
        backup_policy.weeklyBackupLimit = weekly_backup_limit
    if monthly_backup_limit is not None:
        backup_policy.monthlyBackupLimit = monthly_backup_limit
    if description is not None:
        backup_policy.description = description
    if labels is not None:
        backup_policy.labels = labels
    return backup_policy