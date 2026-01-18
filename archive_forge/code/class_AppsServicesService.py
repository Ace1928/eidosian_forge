from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsServicesService(base_api.BaseApiService):
    """Service class for the apps_services resource."""
    _NAME = 'apps_services'

    def __init__(self, client):
        super(AppengineV1beta.AppsServicesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified service and all enclosed versions.

      Args:
        request: (AppengineAppsServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}', http_method='DELETE', method_id='appengine.apps.services.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the current configuration of the specified service.

      Args:
        request: (AppengineAppsServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}', http_method='GET', method_id='appengine.apps.services.get', ordered_params=['name'], path_params=['name'], query_params=['includeExtraData'], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsServicesGetRequest', response_type_name='Service', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the services in the application.

      Args:
        request: (AppengineAppsServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services', http_method='GET', method_id='appengine.apps.services.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/services', request_field='', request_type_name='AppengineAppsServicesListRequest', response_type_name='ListServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the configuration of the specified service.

      Args:
        request: (AppengineAppsServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}', http_method='PATCH', method_id='appengine.apps.services.patch', ordered_params=['name'], path_params=['name'], query_params=['migrateTraffic', 'updateMask'], relative_path='v1beta/{+name}', request_field='service', request_type_name='AppengineAppsServicesPatchRequest', response_type_name='Operation', supports_download=False)