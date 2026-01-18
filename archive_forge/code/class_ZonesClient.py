from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking import utils
class ZonesClient(object):
    """Client for zone resource of GDCE fabric API."""

    def __init__(self, release_track, client=None, messages=None):
        self._client = client or utils.GetClientInstance(release_track)
        self._messages = messages or utils.GetMessagesModule(release_track)
        self._service = self._client.projects_locations_zones

    def InitializeZone(self, zone_ref):
        """Initialzie a specified zone."""
        zone_init_req = self._messages.EdgenetworkProjectsLocationsZonesInitializeRequest(name=zone_ref.RelativeName())
        return self._service.Initialize(zone_init_req)