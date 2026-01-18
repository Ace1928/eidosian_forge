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
def ParseBackupVault(self, name=None, description=None, labels=None):
    """Parses the command line arguments for Create BackupVault into a message.

    Args:
      name: The name of the Backup Vault.
      description: The description of the Backup Vault.
      labels: The parsed labels value.

    Returns:
      The configuration that will be used ass the request body for creating a
      Cloud NetApp Backup Vault.
    """
    backup_vault = self.messages.BackupVault()
    backup_vault.name = name
    backup_vault.description = description
    backup_vault.labels = labels
    return backup_vault