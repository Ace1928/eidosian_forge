from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
class ProjectsLocationsPowerImagesService(base_api.BaseApiService):
    """Service class for the projects_locations_powerImages resource."""
    _NAME = 'projects_locations_powerImages'

    def __init__(self, client):
        super(MarketplacesolutionsV1alpha1.ProjectsLocationsPowerImagesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details about a single image from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerImagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PowerImage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerImages/{powerImagesId}', http_method='GET', method_id='marketplacesolutions.projects.locations.powerImages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerImagesGetRequest', response_type_name='PowerImage', supports_download=False)

    def List(self, request, global_params=None):
        """List Images in a given project and location from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerImagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPowerImagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerImages', http_method='GET', method_id='marketplacesolutions.projects.locations.powerImages.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/powerImages', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerImagesListRequest', response_type_name='ListPowerImagesResponse', supports_download=False)