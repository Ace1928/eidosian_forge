from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
class ProjectsBrandsIdentityAwareProxyClientsService(base_api.BaseApiService):
    """Service class for the projects_brands_identityAwareProxyClients resource."""
    _NAME = 'projects_brands_identityAwareProxyClients'

    def __init__(self, client):
        super(IapV1.ProjectsBrandsIdentityAwareProxyClientsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Identity Aware Proxy (IAP) OAuth client. The client is owned by IAP. Requires that the brand for the project exists and that it is set for internal-only use.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdentityAwareProxyClient) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}/identityAwareProxyClients', http_method='POST', method_id='iap.projects.brands.identityAwareProxyClients.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/identityAwareProxyClients', request_field='identityAwareProxyClient', request_type_name='IapProjectsBrandsIdentityAwareProxyClientsCreateRequest', response_type_name='IdentityAwareProxyClient', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Identity Aware Proxy (IAP) OAuth client. Useful for removing obsolete clients, managing the number of clients in a given project, and cleaning up after tests. Requires that the client is owned by IAP.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}/identityAwareProxyClients/{identityAwareProxyClientsId}', http_method='DELETE', method_id='iap.projects.brands.identityAwareProxyClients.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IapProjectsBrandsIdentityAwareProxyClientsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves an Identity Aware Proxy (IAP) OAuth client. Requires that the client is owned by IAP.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdentityAwareProxyClient) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}/identityAwareProxyClients/{identityAwareProxyClientsId}', http_method='GET', method_id='iap.projects.brands.identityAwareProxyClients.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IapProjectsBrandsIdentityAwareProxyClientsGetRequest', response_type_name='IdentityAwareProxyClient', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing clients for the brand.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListIdentityAwareProxyClientsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}/identityAwareProxyClients', http_method='GET', method_id='iap.projects.brands.identityAwareProxyClients.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/identityAwareProxyClients', request_field='', request_type_name='IapProjectsBrandsIdentityAwareProxyClientsListRequest', response_type_name='ListIdentityAwareProxyClientsResponse', supports_download=False)

    def ResetSecret(self, request, global_params=None):
        """Resets an Identity Aware Proxy (IAP) OAuth client secret. Useful if the secret was compromised. Requires that the client is owned by IAP.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsResetSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdentityAwareProxyClient) The response message.
      """
        config = self.GetMethodConfig('ResetSecret')
        return self._RunMethod(config, request, global_params=global_params)
    ResetSecret.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/brands/{brandsId}/identityAwareProxyClients/{identityAwareProxyClientsId}:resetSecret', http_method='POST', method_id='iap.projects.brands.identityAwareProxyClients.resetSecret', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resetSecret', request_field='resetIdentityAwareProxyClientSecretRequest', request_type_name='IapProjectsBrandsIdentityAwareProxyClientsResetSecretRequest', response_type_name='IdentityAwareProxyClient', supports_download=False)