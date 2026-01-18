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
def ListBackupVaults(self, location_ref, limit=None):
    """Make API calls to List Cloud NetApp Backup Vaults.

    Args:
      location_ref: The parsed location of the listed NetApp Backup Vaults.
      limit: The number of Cloud NetApp Backup Vaults to limit the results to.
        This limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp Backup Vaults.
    """
    request = self.messages.NetappProjectsLocationsBackupVaultsListRequest(parent=location_ref)
    response = self.client.projects_locations_backupVaults.List(request)
    for location in response.unreachable:
        log.warning('Location {} may be unreachable.'.format(location))
    return list_pager.YieldFromList(self.client.projects_locations_backupVaults, request, field=constants.BACKUP_VAULT_RESOURCE, limit=limit, batch_size_attribute='pageSize')