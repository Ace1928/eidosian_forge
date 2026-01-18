from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsVolumesLunsService(base_api.BaseApiService):
    """Service class for the projects_locations_volumes_luns resource."""
    _NAME = 'projects_locations_volumes_luns'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsVolumesLunsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete a Lun. Lun shouldn't be attached to any Instances.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesLunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/luns/{lunsId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.volumes.luns.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesLunsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Evict(self, request, global_params=None):
        """Skips lun's cooloff and deletes it now. Lun must be in cooloff state.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesLunsEvictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Evict')
        return self._RunMethod(config, request, global_params=global_params)
    Evict.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/luns/{lunsId}:evict', http_method='POST', method_id='baremetalsolution.projects.locations.volumes.luns.evict', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:evict', request_field='evictLunRequest', request_type_name='BaremetalsolutionProjectsLocationsVolumesLunsEvictRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details of a single storage logical unit number(LUN).

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesLunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Lun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/luns/{lunsId}', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.luns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesLunsGetRequest', response_type_name='Lun', supports_download=False)

    def List(self, request, global_params=None):
        """List storage volume luns for given storage volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesLunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/volumes/{volumesId}/luns', http_method='GET', method_id='baremetalsolution.projects.locations.volumes.luns.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/luns', request_field='', request_type_name='BaremetalsolutionProjectsLocationsVolumesLunsListRequest', response_type_name='ListLunsResponse', supports_download=False)