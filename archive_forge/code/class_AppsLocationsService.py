from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsLocationsService(base_api.BaseApiService):
    """Service class for the apps_locations resource."""
    _NAME = 'apps_locations'

    def __init__(self, client):
        super(AppengineV1beta.AppsLocationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets information about a location.

      Args:
        request: (AppengineAppsLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Location) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/locations/{locationsId}', http_method='GET', method_id='appengine.apps.locations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsLocationsGetRequest', response_type_name='Location', supports_download=False)

    def List(self, request, global_params=None):
        """Lists information about the supported locations for this service.

      Args:
        request: (AppengineAppsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/locations', http_method='GET', method_id='appengine.apps.locations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+name}/locations', request_field='', request_type_name='AppengineAppsLocationsListRequest', response_type_name='ListLocationsResponse', supports_download=False)