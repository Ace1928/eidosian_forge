from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsPartnerSSEEnvironmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_partnerSSEEnvironments resource."""
    _NAME = 'projects_locations_partnerSSEEnvironments'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsPartnerSSEEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new PartnerSSEEnvironment in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEEnvironmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEEnvironments', http_method='POST', method_id='networksecurity.projects.locations.partnerSSEEnvironments.create', ordered_params=['parent'], path_params=['parent'], query_params=['partnerSseEnvironmentId', 'requestId'], relative_path='v1alpha1/{+parent}/partnerSSEEnvironments', request_field='partnerSSEEnvironment', request_type_name='NetworksecurityProjectsLocationsPartnerSSEEnvironmentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single PartnerSSEEnvironment.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEEnvironmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEEnvironments/{partnerSSEEnvironmentsId}', http_method='DELETE', method_id='networksecurity.projects.locations.partnerSSEEnvironments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEEnvironmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single PartnerSSEEnvironment.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartnerSSEEnvironment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEEnvironments/{partnerSSEEnvironmentsId}', http_method='GET', method_id='networksecurity.projects.locations.partnerSSEEnvironments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEEnvironmentsGetRequest', response_type_name='PartnerSSEEnvironment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PartnerSSEEnvironments in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEEnvironmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPartnerSSEEnvironmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEEnvironments', http_method='GET', method_id='networksecurity.projects.locations.partnerSSEEnvironments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/partnerSSEEnvironments', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEEnvironmentsListRequest', response_type_name='ListPartnerSSEEnvironmentsResponse', supports_download=False)