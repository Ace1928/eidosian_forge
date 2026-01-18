from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsOsImagesService(base_api.BaseApiService):
    """Service class for the projects_locations_osImages resource."""
    _NAME = 'projects_locations_osImages'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsOsImagesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details of a single OS image.

      Args:
        request: (BaremetalsolutionProjectsLocationsOsImagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OSImage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/osImages/{osImagesId}', http_method='GET', method_id='baremetalsolution.projects.locations.osImages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsOsImagesGetRequest', response_type_name='OSImage', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of OS images which are currently approved.

      Args:
        request: (BaremetalsolutionProjectsLocationsOsImagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOSImagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/osImages', http_method='GET', method_id='baremetalsolution.projects.locations.osImages.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/osImages', request_field='', request_type_name='BaremetalsolutionProjectsLocationsOsImagesListRequest', response_type_name='ListOSImagesResponse', supports_download=False)