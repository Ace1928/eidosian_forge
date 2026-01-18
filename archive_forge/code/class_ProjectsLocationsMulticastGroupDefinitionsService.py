from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastGroupDefinitionsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastGroupDefinitions resource."""
    _NAME = 'projects_locations_multicastGroupDefinitions'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastGroupDefinitionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast group definition in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupDefinitionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupDefinitions', http_method='POST', method_id='networkservices.projects.locations.multicastGroupDefinitions.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastGroupDefinitionId', 'requestId'], relative_path='v1/{+parent}/multicastGroupDefinitions', request_field='multicastGroupDefinition', request_type_name='NetworkservicesProjectsLocationsMulticastGroupDefinitionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast group definition.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupDefinitionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupDefinitions/{multicastGroupDefinitionsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastGroupDefinitions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupDefinitionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast group definition.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupDefinitionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastGroupDefinition) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupDefinitions/{multicastGroupDefinitionsId}', http_method='GET', method_id='networkservices.projects.locations.multicastGroupDefinitions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupDefinitionsGetRequest', response_type_name='MulticastGroupDefinition', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast group definitions in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupDefinitionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastGroupDefinitionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupDefinitions', http_method='GET', method_id='networkservices.projects.locations.multicastGroupDefinitions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastGroupDefinitions', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupDefinitionsListRequest', response_type_name='ListMulticastGroupDefinitionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast group definition.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupDefinitionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupDefinitions/{multicastGroupDefinitionsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastGroupDefinitions.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastGroupDefinition', request_type_name='NetworkservicesProjectsLocationsMulticastGroupDefinitionsPatchRequest', response_type_name='Operation', supports_download=False)