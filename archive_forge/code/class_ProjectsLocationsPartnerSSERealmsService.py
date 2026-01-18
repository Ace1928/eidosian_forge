from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsPartnerSSERealmsService(base_api.BaseApiService):
    """Service class for the projects_locations_partnerSSERealms resource."""
    _NAME = 'projects_locations_partnerSSERealms'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsPartnerSSERealmsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new PartnerSSERealm in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSERealmsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSERealms', http_method='POST', method_id='networksecurity.projects.locations.partnerSSERealms.create', ordered_params=['parent'], path_params=['parent'], query_params=['partnerSseRealmId', 'requestId'], relative_path='v1alpha1/{+parent}/partnerSSERealms', request_field='partnerSSERealm', request_type_name='NetworksecurityProjectsLocationsPartnerSSERealmsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single PartnerSSERealm.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSERealmsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSERealms/{partnerSSERealmsId}', http_method='DELETE', method_id='networksecurity.projects.locations.partnerSSERealms.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSERealmsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single PartnerSSERealm.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSERealmsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartnerSSERealm) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSERealms/{partnerSSERealmsId}', http_method='GET', method_id='networksecurity.projects.locations.partnerSSERealms.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSERealmsGetRequest', response_type_name='PartnerSSERealm', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PartnerSSERealms in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSERealmsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPartnerSSERealmsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSERealms', http_method='GET', method_id='networksecurity.projects.locations.partnerSSERealms.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/partnerSSERealms', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSERealmsListRequest', response_type_name='ListPartnerSSERealmsResponse', supports_download=False)