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
def ListBackupPolicies(self, location_ref, limit=None):
    """Make API calls to List Cloud NetApp Backup Policies.

    Args:
      location_ref: The parsed location of the listed NetApp Backup Policies.
      limit: The number of Cloud NetApp Backup Policies
        to limit the results to. This limit is passed to
        the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp Backup Policies.
    """
    request = self.messages.NetappProjectsLocationsBackupPoliciesListRequest(parent=location_ref)
    response = self.client.projects_locations_backupPolicies.List(request)
    for location in response.unreachable:
        log.warning('Location {} may be unreachable.'.format(location))
    return list_pager.YieldFromList(self.client.projects_locations_backupPolicies, request, field=constants.BACKUP_POLICY_RESOURCE, limit=limit, batch_size_attribute='pageSize')