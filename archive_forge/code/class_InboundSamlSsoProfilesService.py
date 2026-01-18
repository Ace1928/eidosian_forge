from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class InboundSamlSsoProfilesService(base_api.BaseApiService):
    """Service class for the inboundSamlSsoProfiles resource."""
    _NAME = 'inboundSamlSsoProfiles'

    def __init__(self, client):
        super(CloudidentityV1.InboundSamlSsoProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an InboundSamlSsoProfile for a customer.

      Args:
        request: (InboundSamlSsoProfile) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudidentity.inboundSamlSsoProfiles.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1/inboundSamlSsoProfiles', request_field='<request>', request_type_name='InboundSamlSsoProfile', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an InboundSamlSsoProfile.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}', http_method='DELETE', method_id='cloudidentity.inboundSamlSsoProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an InboundSamlSsoProfile.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InboundSamlSsoProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}', http_method='GET', method_id='cloudidentity.inboundSamlSsoProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesGetRequest', response_type_name='InboundSamlSsoProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists InboundSamlSsoProfiles for a customer.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInboundSamlSsoProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.inboundSamlSsoProfiles.list', ordered_params=[], path_params=[], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/inboundSamlSsoProfiles', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesListRequest', response_type_name='ListInboundSamlSsoProfilesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an InboundSamlSsoProfile.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}', http_method='PATCH', method_id='cloudidentity.inboundSamlSsoProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='inboundSamlSsoProfile', request_type_name='CloudidentityInboundSamlSsoProfilesPatchRequest', response_type_name='Operation', supports_download=False)