from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsUrlListsService(base_api.BaseApiService):
    """Service class for the projects_locations_urlLists resource."""
    _NAME = 'projects_locations_urlLists'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsUrlListsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new UrlList in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsUrlListsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/urlLists', http_method='POST', method_id='networksecurity.projects.locations.urlLists.create', ordered_params=['parent'], path_params=['parent'], query_params=['urlListId'], relative_path='v1/{+parent}/urlLists', request_field='urlList', request_type_name='NetworksecurityProjectsLocationsUrlListsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single UrlList.

      Args:
        request: (NetworksecurityProjectsLocationsUrlListsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/urlLists/{urlListsId}', http_method='DELETE', method_id='networksecurity.projects.locations.urlLists.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsUrlListsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single UrlList.

      Args:
        request: (NetworksecurityProjectsLocationsUrlListsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlList) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/urlLists/{urlListsId}', http_method='GET', method_id='networksecurity.projects.locations.urlLists.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsUrlListsGetRequest', response_type_name='UrlList', supports_download=False)

    def List(self, request, global_params=None):
        """Lists UrlLists in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsUrlListsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUrlListsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/urlLists', http_method='GET', method_id='networksecurity.projects.locations.urlLists.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/urlLists', request_field='', request_type_name='NetworksecurityProjectsLocationsUrlListsListRequest', response_type_name='ListUrlListsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single UrlList.

      Args:
        request: (NetworksecurityProjectsLocationsUrlListsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/urlLists/{urlListsId}', http_method='PATCH', method_id='networksecurity.projects.locations.urlLists.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='urlList', request_type_name='NetworksecurityProjectsLocationsUrlListsPatchRequest', response_type_name='Operation', supports_download=False)