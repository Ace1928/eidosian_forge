from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudnumberregistry.v1alpha import cloudnumberregistry_v1alpha_messages as messages
class ProjectsLocationsRegistryBooksRegistryNodesService(base_api.BaseApiService):
    """Service class for the projects_locations_registryBooks_registryNodes resource."""
    _NAME = 'projects_locations_registryBooks_registryNodes'

    def __init__(self, client):
        super(CloudnumberregistryV1alpha.ProjectsLocationsRegistryBooksRegistryNodesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new RegistryNode in a given project and location.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/registryNodes', http_method='POST', method_id='cloudnumberregistry.projects.locations.registryBooks.registryNodes.create', ordered_params=['parent'], path_params=['parent'], query_params=['registryNodeId', 'requestId'], relative_path='v1alpha/{+parent}/registryNodes', request_field='registryNode', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single RegistryNode.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/registryNodes/{registryNodesId}', http_method='DELETE', method_id='cloudnumberregistry.projects.locations.registryBooks.registryNodes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single RegistryNode.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegistryNode) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/registryNodes/{registryNodesId}', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.registryNodes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesGetRequest', response_type_name='RegistryNode', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RegistryNodes in a given project and location.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRegistryNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/registryNodes', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.registryNodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/registryNodes', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesListRequest', response_type_name='ListRegistryNodesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single RegistryNode.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/registryNodes/{registryNodesId}', http_method='PATCH', method_id='cloudnumberregistry.projects.locations.registryBooks.registryNodes.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='registryNode', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksRegistryNodesPatchRequest', response_type_name='Operation', supports_download=False)