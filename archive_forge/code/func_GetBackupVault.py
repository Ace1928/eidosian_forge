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
def GetBackupVault(self, backupvault_ref):
    """Get Cloud NetApp Backup Vault information."""
    request = self.messages.NetappProjectsLocationsBackupVaultsGetRequest(name=backupvault_ref.RelativeName())
    return self.client.projects_locations_backupVaults.Get(request)