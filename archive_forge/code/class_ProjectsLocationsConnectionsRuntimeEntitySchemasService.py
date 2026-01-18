from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsConnectionsRuntimeEntitySchemasService(base_api.BaseApiService):
    """Service class for the projects_locations_connections_runtimeEntitySchemas resource."""
    _NAME = 'projects_locations_connections_runtimeEntitySchemas'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsConnectionsRuntimeEntitySchemasService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List schema of a runtime entities filtered by entity name.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsRuntimeEntitySchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimeEntitySchemasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/runtimeEntitySchemas', http_method='GET', method_id='connectors.projects.locations.connections.runtimeEntitySchemas.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/runtimeEntitySchemas', request_field='', request_type_name='ConnectorsProjectsLocationsConnectionsRuntimeEntitySchemasListRequest', response_type_name='ListRuntimeEntitySchemasResponse', supports_download=False)