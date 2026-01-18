from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsService(base_api.BaseApiService):
    """Service class for the apps resource."""
    _NAME = 'apps'

    def __init__(self, client):
        super(AppengineV1beta.AppsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an App Engine application for a Google Cloud Platform project. Required fields: id - The ID of the target Cloud Platform project. location - The region (https://cloud.google.com/appengine/docs/locations) where you want the App Engine application located.For more information about App Engine applications, see Managing Projects, Applications, and Billing (https://cloud.google.com/appengine/docs/standard/python/console/).

      Args:
        request: (Application) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='appengine.apps.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1beta/apps', request_field='<request>', request_type_name='Application', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about an application.

      Args:
        request: (AppengineAppsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Application) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}', http_method='GET', method_id='appengine.apps.get', ordered_params=['name'], path_params=['name'], query_params=['includeExtraData'], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsGetRequest', response_type_name='Application', supports_download=False)

    def ListRuntimes(self, request, global_params=None):
        """Lists all the available runtimes for the application.

      Args:
        request: (AppengineAppsListRuntimesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimesResponse) The response message.
      """
        config = self.GetMethodConfig('ListRuntimes')
        return self._RunMethod(config, request, global_params=global_params)
    ListRuntimes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}:listRuntimes', http_method='GET', method_id='appengine.apps.listRuntimes', ordered_params=['parent'], path_params=['parent'], query_params=['environment'], relative_path='v1beta/{+parent}:listRuntimes', request_field='', request_type_name='AppengineAppsListRuntimesRequest', response_type_name='ListRuntimesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified Application resource. You can update the following fields: auth_domain - Google authentication domain for controlling user access to the application. default_cookie_expiration - Cookie expiration policy for the application. iap - Identity-Aware Proxy properties for the application.

      Args:
        request: (AppengineAppsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}', http_method='PATCH', method_id='appengine.apps.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='application', request_type_name='AppengineAppsPatchRequest', response_type_name='Operation', supports_download=False)

    def Repair(self, request, global_params=None):
        """Recreates the required App Engine features for the specified App Engine application, for example a Cloud Storage bucket or App Engine service account. Use this method if you receive an error message about a missing feature, for example, Error retrieving the App Engine service account. If you have deleted your App Engine service account, this will not be able to recreate it. Instead, you should attempt to use the IAM undelete API if possible at https://cloud.google.com/iam/reference/rest/v1/projects.serviceAccounts/undelete?apix_params=%7B"name"%3A"projects%2F-%2FserviceAccounts%2Funique_id"%2C"resource"%3A%7B%7D%7D . If the deletion was recent, the numeric ID can be found in the Cloud Console Activity Log.

      Args:
        request: (AppengineAppsRepairRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Repair')
        return self._RunMethod(config, request, global_params=global_params)
    Repair.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}:repair', http_method='POST', method_id='appengine.apps.repair', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:repair', request_field='repairApplicationRequest', request_type_name='AppengineAppsRepairRequest', response_type_name='Operation', supports_download=False)