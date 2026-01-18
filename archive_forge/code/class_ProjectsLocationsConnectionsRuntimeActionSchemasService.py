from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsConnectionsRuntimeActionSchemasService(base_api.BaseApiService):
    """Service class for the projects_locations_connections_runtimeActionSchemas resource."""
    _NAME = 'projects_locations_connections_runtimeActionSchemas'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsConnectionsRuntimeActionSchemasService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List schema of a runtime actions filtered by action name.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsRuntimeActionSchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimeActionSchemasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/runtimeActionSchemas', http_method='GET', method_id='connectors.projects.locations.connections.runtimeActionSchemas.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/runtimeActionSchemas', request_field='', request_type_name='ConnectorsProjectsLocationsConnectionsRuntimeActionSchemasListRequest', response_type_name='ListRuntimeActionSchemasResponse', supports_download=False)