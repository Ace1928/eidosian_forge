from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsConnectionsConnectionSchemaMetadataService(base_api.BaseApiService):
    """Service class for the projects_locations_connections_connectionSchemaMetadata resource."""
    _NAME = 'projects_locations_connections_connectionSchemaMetadata'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsConnectionsConnectionSchemaMetadataService, self).__init__(client)
        self._upload_configs = {}

    def Refresh(self, request, global_params=None):
        """Refresh runtime schema of a connection.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsConnectionSchemaMetadataRefreshRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Refresh')
        return self._RunMethod(config, request, global_params=global_params)
    Refresh.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/connectionSchemaMetadata:refresh', http_method='POST', method_id='connectors.projects.locations.connections.connectionSchemaMetadata.refresh', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:refresh', request_field='refreshConnectionSchemaMetadataRequest', request_type_name='ConnectorsProjectsLocationsConnectionsConnectionSchemaMetadataRefreshRequest', response_type_name='Operation', supports_download=False)