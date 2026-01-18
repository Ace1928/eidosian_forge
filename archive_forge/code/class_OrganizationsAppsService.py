from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsAppsService(base_api.BaseApiService):
    """Service class for the organizations_apps resource."""
    _NAME = 'organizations_apps'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsAppsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the app profile for the specified app ID.

      Args:
        request: (ApigeeOrganizationsAppsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1App) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apps/{appsId}', http_method='GET', method_id='apigee.organizations.apps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsAppsGetRequest', response_type_name='GoogleCloudApigeeV1App', supports_download=False)

    def List(self, request, global_params=None):
        """Lists IDs of apps within an organization that have the specified app status (approved or revoked) or are of the specified app type (developer or company).

      Args:
        request: (ApigeeOrganizationsAppsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListAppsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apps', http_method='GET', method_id='apigee.organizations.apps.list', ordered_params=['parent'], path_params=['parent'], query_params=['apiProduct', 'apptype', 'expand', 'filter', 'ids', 'includeCred', 'keyStatus', 'pageSize', 'pageToken', 'rows', 'startKey', 'status'], relative_path='v1/{+parent}/apps', request_field='', request_type_name='ApigeeOrganizationsAppsListRequest', response_type_name='GoogleCloudApigeeV1ListAppsResponse', supports_download=False)