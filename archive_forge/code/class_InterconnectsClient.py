from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking import utils
class InterconnectsClient(object):
    """Client for private connections service in the API."""

    def __init__(self, release_track, client=None, messages=None):
        self._client = client or utils.GetClientInstance(release_track)
        self._messages = messages or utils.GetMessagesModule(release_track)
        self._service = self._client.projects_locations_zones_interconnects

    def GetStatus(self, interconnect_ref):
        """Get the status of a specified interconnect."""
        get_interconnect_status_req = self._messages.EdgenetworkProjectsLocationsZonesInterconnectsDiagnoseRequest(name=interconnect_ref.RelativeName())
        return self._service.Diagnose(get_interconnect_status_req)