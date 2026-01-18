from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsMuteConfigsService(base_api.BaseApiService):
    """Service class for the organizations_muteConfigs resource."""
    _NAME = 'organizations_muteConfigs'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsMuteConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a mute config.

      Args:
        request: (SecuritycenterOrganizationsMuteConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2MuteConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/muteConfigs', http_method='POST', method_id='securitycenter.organizations.muteConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['muteConfigId'], relative_path='v2/{+parent}/muteConfigs', request_field='googleCloudSecuritycenterV2MuteConfig', request_type_name='SecuritycenterOrganizationsMuteConfigsCreateRequest', response_type_name='GoogleCloudSecuritycenterV2MuteConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing mute config. If no location is specified, default is global.

      Args:
        request: (SecuritycenterOrganizationsMuteConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/muteConfigs/{muteConfigsId}', http_method='DELETE', method_id='securitycenter.organizations.muteConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsMuteConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a mute config. If no location is specified, default is global.

      Args:
        request: (SecuritycenterOrganizationsMuteConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2MuteConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/muteConfigs/{muteConfigsId}', http_method='GET', method_id='securitycenter.organizations.muteConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsMuteConfigsGetRequest', response_type_name='GoogleCloudSecuritycenterV2MuteConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists mute configs. If no location is specified, default is global.

      Args:
        request: (SecuritycenterOrganizationsMuteConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMuteConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/muteConfigs', http_method='GET', method_id='securitycenter.organizations.muteConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/muteConfigs', request_field='', request_type_name='SecuritycenterOrganizationsMuteConfigsListRequest', response_type_name='ListMuteConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a mute config. If no location is specified, default is global.

      Args:
        request: (SecuritycenterOrganizationsMuteConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2MuteConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/muteConfigs/{muteConfigsId}', http_method='PATCH', method_id='securitycenter.organizations.muteConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2MuteConfig', request_type_name='SecuritycenterOrganizationsMuteConfigsPatchRequest', response_type_name='GoogleCloudSecuritycenterV2MuteConfig', supports_download=False)