from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
class OrganizationsOsConfigsService(base_api.BaseApiService):
    """Service class for the organizations_osConfigs resource."""
    _NAME = 'organizations_osConfigs'

    def __init__(self, client):
        super(OsconfigV1alpha1.OrganizationsOsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OsConfig.

      Args:
        request: (OsconfigOrganizationsOsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/osConfigs', http_method='POST', method_id='osconfig.organizations.osConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/osConfigs', request_field='osConfig', request_type_name='OsconfigOrganizationsOsConfigsCreateRequest', response_type_name='OsConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OsConfig.

      Args:
        request: (OsconfigOrganizationsOsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/osConfigs/{osConfigsId}', http_method='DELETE', method_id='osconfig.organizations.osConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigOrganizationsOsConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OsConfig.

      Args:
        request: (OsconfigOrganizationsOsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/osConfigs/{osConfigsId}', http_method='GET', method_id='osconfig.organizations.osConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigOrganizationsOsConfigsGetRequest', response_type_name='OsConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OsConfigs.

      Args:
        request: (OsconfigOrganizationsOsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOsConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/osConfigs', http_method='GET', method_id='osconfig.organizations.osConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/osConfigs', request_field='', request_type_name='OsconfigOrganizationsOsConfigsListRequest', response_type_name='ListOsConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OsConfig.

      Args:
        request: (OsconfigOrganizationsOsConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/osConfigs/{osConfigsId}', http_method='PATCH', method_id='osconfig.organizations.osConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='osConfig', request_type_name='OsconfigOrganizationsOsConfigsPatchRequest', response_type_name='OsConfig', supports_download=False)