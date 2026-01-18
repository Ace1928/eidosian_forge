from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class InboundSamlSsoProfilesIdpCredentialsService(base_api.BaseApiService):
    """Service class for the inboundSamlSsoProfiles_idpCredentials resource."""
    _NAME = 'inboundSamlSsoProfiles_idpCredentials'

    def __init__(self, client):
        super(CloudidentityV1.InboundSamlSsoProfilesIdpCredentialsService, self).__init__(client)
        self._upload_configs = {}

    def Add(self, request, global_params=None):
        """Adds an IdpCredential. Up to 2 credentials are allowed.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesIdpCredentialsAddRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Add')
        return self._RunMethod(config, request, global_params=global_params)
    Add.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}/idpCredentials:add', http_method='POST', method_id='cloudidentity.inboundSamlSsoProfiles.idpCredentials.add', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/idpCredentials:add', request_field='addIdpCredentialRequest', request_type_name='CloudidentityInboundSamlSsoProfilesIdpCredentialsAddRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an IdpCredential.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesIdpCredentialsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}/idpCredentials/{idpCredentialsId}', http_method='DELETE', method_id='cloudidentity.inboundSamlSsoProfiles.idpCredentials.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesIdpCredentialsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an IdpCredential.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesIdpCredentialsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdpCredential) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}/idpCredentials/{idpCredentialsId}', http_method='GET', method_id='cloudidentity.inboundSamlSsoProfiles.idpCredentials.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesIdpCredentialsGetRequest', response_type_name='IdpCredential', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of IdpCredentials in an InboundSamlSsoProfile.

      Args:
        request: (CloudidentityInboundSamlSsoProfilesIdpCredentialsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListIdpCredentialsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSamlSsoProfiles/{inboundSamlSsoProfilesId}/idpCredentials', http_method='GET', method_id='cloudidentity.inboundSamlSsoProfiles.idpCredentials.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/idpCredentials', request_field='', request_type_name='CloudidentityInboundSamlSsoProfilesIdpCredentialsListRequest', response_type_name='ListIdpCredentialsResponse', supports_download=False)