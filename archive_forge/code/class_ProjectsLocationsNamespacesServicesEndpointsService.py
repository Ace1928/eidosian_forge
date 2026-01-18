from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicedirectory.v1 import servicedirectory_v1_messages as messages
class ProjectsLocationsNamespacesServicesEndpointsService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_services_endpoints resource."""
    _NAME = 'projects_locations_namespaces_services_endpoints'

    def __init__(self, client):
        super(ServicedirectoryV1.ProjectsLocationsNamespacesServicesEndpointsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an endpoint, and returns the new endpoint.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesEndpointsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Endpoint) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}/endpoints', http_method='POST', method_id='servicedirectory.projects.locations.namespaces.services.endpoints.create', ordered_params=['parent'], path_params=['parent'], query_params=['endpointId'], relative_path='v1/{+parent}/endpoints', request_field='endpoint', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesEndpointsCreateRequest', response_type_name='Endpoint', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an endpoint.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesEndpointsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}/endpoints/{endpointsId}', http_method='DELETE', method_id='servicedirectory.projects.locations.namespaces.services.endpoints.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesEndpointsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an endpoint.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesEndpointsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Endpoint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}/endpoints/{endpointsId}', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.services.endpoints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesEndpointsGetRequest', response_type_name='Endpoint', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all endpoints.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesEndpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEndpointsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}/endpoints', http_method='GET', method_id='servicedirectory.projects.locations.namespaces.services.endpoints.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/endpoints', request_field='', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesEndpointsListRequest', response_type_name='ListEndpointsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an endpoint.

      Args:
        request: (ServicedirectoryProjectsLocationsNamespacesServicesEndpointsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Endpoint) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/services/{servicesId}/endpoints/{endpointsId}', http_method='PATCH', method_id='servicedirectory.projects.locations.namespaces.services.endpoints.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='endpoint', request_type_name='ServicedirectoryProjectsLocationsNamespacesServicesEndpointsPatchRequest', response_type_name='Endpoint', supports_download=False)