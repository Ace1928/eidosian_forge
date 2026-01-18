from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
class ProjectsIapTunnelLocationsService(base_api.BaseApiService):
    """Service class for the projects_iap_tunnel_locations resource."""
    _NAME = 'projects_iap_tunnel_locations'

    def __init__(self, client):
        super(IapV1.ProjectsIapTunnelLocationsService, self).__init__(client)
        self._upload_configs = {}