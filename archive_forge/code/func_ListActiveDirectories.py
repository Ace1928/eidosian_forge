from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ListActiveDirectories(self, location_ref, limit=None):
    """Make API calls to List active Cloud NetApp Active Directories.

    Args:
      location_ref: The parsed location of the listed NetApp Active Directories.
      limit: The number of Cloud NetApp Active Directories
        to limit the results to. This limit is passed to
        the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp Active Directories.
    """
    request = self.messages.NetappProjectsLocationsActiveDirectoriesListRequest(parent=location_ref)
    response = self.client.projects_locations_activeDirectories.List(request)
    for location in response.unreachable:
        log.warning('Location {} may be unreachable.'.format(location))
    return list_pager.YieldFromList(self.client.projects_locations_activeDirectories, request, field=constants.ACTIVE_DIRECTORY_RESOURCE, limit=limit, batch_size_attribute='pageSize')