from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsLbTrafficExtensionsService(base_api.BaseApiService):
    """Service class for the projects_locations_lbTrafficExtensions resource."""
    _NAME = 'projects_locations_lbTrafficExtensions'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsLbTrafficExtensionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `LbTrafficExtension` resource in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsLbTrafficExtensionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lbTrafficExtensions', http_method='POST', method_id='networkservices.projects.locations.lbTrafficExtensions.create', ordered_params=['parent'], path_params=['parent'], query_params=['lbTrafficExtensionId', 'requestId'], relative_path='v1/{+parent}/lbTrafficExtensions', request_field='lbTrafficExtension', request_type_name='NetworkservicesProjectsLocationsLbTrafficExtensionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified `LbTrafficExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbTrafficExtensionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lbTrafficExtensions/{lbTrafficExtensionsId}', http_method='DELETE', method_id='networkservices.projects.locations.lbTrafficExtensions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsLbTrafficExtensionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of the specified `LbTrafficExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbTrafficExtensionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LbTrafficExtension) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lbTrafficExtensions/{lbTrafficExtensionsId}', http_method='GET', method_id='networkservices.projects.locations.lbTrafficExtensions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsLbTrafficExtensionsGetRequest', response_type_name='LbTrafficExtension', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `LbTrafficExtension` resources in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsLbTrafficExtensionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLbTrafficExtensionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lbTrafficExtensions', http_method='GET', method_id='networkservices.projects.locations.lbTrafficExtensions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/lbTrafficExtensions', request_field='', request_type_name='NetworkservicesProjectsLocationsLbTrafficExtensionsListRequest', response_type_name='ListLbTrafficExtensionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of the specified `LbTrafficExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbTrafficExtensionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lbTrafficExtensions/{lbTrafficExtensionsId}', http_method='PATCH', method_id='networkservices.projects.locations.lbTrafficExtensions.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='lbTrafficExtension', request_type_name='NetworkservicesProjectsLocationsLbTrafficExtensionsPatchRequest', response_type_name='Operation', supports_download=False)