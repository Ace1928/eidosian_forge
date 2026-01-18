from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking import utils
class NetworksClient(object):
    """Client for network resource of GDCE fabric API."""

    def __init__(self, release_track, client=None, messages=None):
        self._client = client or utils.GetClientInstance(release_track)
        self._messages = messages or utils.GetMessagesModule(release_track)
        self._service = self._client.projects_locations_zones_networks

    def GetStatus(self, network_ref):
        """Get the status of a specified network."""
        get_network_status_req = self._messages.EdgenetworkProjectsLocationsZonesNetworksDiagnoseRequest(name=network_ref.RelativeName())
        return self._service.Diagnose(get_network_status_req)