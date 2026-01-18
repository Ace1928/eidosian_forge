from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateConnectionsPeeringRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_privateConnections_peeringRoutes resource."""
    _NAME = 'projects_locations_privateConnections_peeringRoutes'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateConnectionsPeeringRoutesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the private connection routes exchanged over a peering connection.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsPeeringRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPrivateConnectionPeeringRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}/peeringRoutes', http_method='GET', method_id='vmwareengine.projects.locations.privateConnections.peeringRoutes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/peeringRoutes', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsPeeringRoutesListRequest', response_type_name='ListPrivateConnectionPeeringRoutesResponse', supports_download=False)