from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkmanagement.v1alpha1 import networkmanagement_v1alpha1_messages as messages
class ProjectsLocationsAppliancesService(base_api.BaseApiService):
    """Service class for the projects_locations_appliances resource."""
    _NAME = 'projects_locations_appliances'

    def __init__(self, client):
        super(NetworkmanagementV1alpha1.ProjectsLocationsAppliancesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single Appliance.

      Args:
        request: (NetworkmanagementProjectsLocationsAppliancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Appliance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/appliances/{appliancesId}', http_method='GET', method_id='networkmanagement.projects.locations.appliances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsAppliancesGetRequest', response_type_name='Appliance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists available third party appliance resources.

      Args:
        request: (NetworkmanagementProjectsLocationsAppliancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAppliancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/appliances', http_method='GET', method_id='networkmanagement.projects.locations.appliances.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/appliances', request_field='', request_type_name='NetworkmanagementProjectsLocationsAppliancesListRequest', response_type_name='ListAppliancesResponse', supports_download=False)