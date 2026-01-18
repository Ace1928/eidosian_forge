from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_api
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_connectivity import flags
from googlecloudsdk.command_lib.network_connectivity import util
def ViewToEnum(view, release_track):
    """Converts the typed view into its Enum value."""
    list_hub_spokes_req = networkconnectivity_util.GetMessagesModule(release_track).NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest
    if view == 'detailed':
        return list_hub_spokes_req.ViewValueValuesEnum.DETAILED
    return list_hub_spokes_req.ViewValueValuesEnum.BASIC