from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2 import tpu_v2_messages as messages
class ProjectsLocationsAcceleratorTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_acceleratorTypes resource."""
    _NAME = 'projects_locations_acceleratorTypes'

    def __init__(self, client):
        super(TpuV2.ProjectsLocationsAcceleratorTypesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets AcceleratorType.

      Args:
        request: (TpuProjectsLocationsAcceleratorTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AcceleratorType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/acceleratorTypes/{acceleratorTypesId}', http_method='GET', method_id='tpu.projects.locations.acceleratorTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsAcceleratorTypesGetRequest', response_type_name='AcceleratorType', supports_download=False)

    def List(self, request, global_params=None):
        """Lists accelerator types supported by this API.

      Args:
        request: (TpuProjectsLocationsAcceleratorTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAcceleratorTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/acceleratorTypes', http_method='GET', method_id='tpu.projects.locations.acceleratorTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/acceleratorTypes', request_field='', request_type_name='TpuProjectsLocationsAcceleratorTypesListRequest', response_type_name='ListAcceleratorTypesResponse', supports_download=False)