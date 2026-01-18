from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsInstancesInventoriesService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_inventories resource."""
    _NAME = 'projects_locations_instances_inventories'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsInstancesInventoriesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get inventory data for the specified VM instance. If the VM has no associated inventory, the message `NOT_FOUND` is returned.

      Args:
        request: (OsconfigProjectsLocationsInstancesInventoriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Inventory) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/inventory', http_method='GET', method_id='osconfig.projects.locations.instances.inventories.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsInstancesInventoriesGetRequest', response_type_name='Inventory', supports_download=False)

    def List(self, request, global_params=None):
        """List inventory data for all VM instances in the specified zone.

      Args:
        request: (OsconfigProjectsLocationsInstancesInventoriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInventoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/inventories', http_method='GET', method_id='osconfig.projects.locations.instances.inventories.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'view'], relative_path='v1alpha/{+parent}/inventories', request_field='', request_type_name='OsconfigProjectsLocationsInstancesInventoriesListRequest', response_type_name='ListInventoriesResponse', supports_download=False)