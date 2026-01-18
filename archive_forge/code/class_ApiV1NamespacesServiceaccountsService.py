from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ApiV1NamespacesServiceaccountsService(base_api.BaseApiService):
    """Service class for the api_v1_namespaces_serviceaccounts resource."""
    _NAME = 'api_v1_namespaces_serviceaccounts'

    def __init__(self, client):
        super(AnthoseventsV1.ApiV1NamespacesServiceaccountsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new service account.

      Args:
        request: (AnthoseventsApiV1NamespacesServiceaccountsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/serviceaccounts', http_method='POST', method_id='anthosevents.api.v1.namespaces.serviceaccounts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='api/v1/{+parent}/serviceaccounts', request_field='serviceAccount', request_type_name='AnthoseventsApiV1NamespacesServiceaccountsCreateRequest', response_type_name='ServiceAccount', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to retrieve service account.

      Args:
        request: (AnthoseventsApiV1NamespacesServiceaccountsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/serviceaccounts/{serviceaccountsId}', http_method='GET', method_id='anthosevents.api.v1.namespaces.serviceaccounts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='', request_type_name='AnthoseventsApiV1NamespacesServiceaccountsGetRequest', response_type_name='ServiceAccount', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list Service Accounts.

      Args:
        request: (AnthoseventsApiV1NamespacesServiceaccountsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceAccountsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/serviceaccounts', http_method='GET', method_id='anthosevents.api.v1.namespaces.serviceaccounts.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='api/v1/{+parent}/serviceaccounts', request_field='', request_type_name='AnthoseventsApiV1NamespacesServiceaccountsListRequest', response_type_name='ListServiceAccountsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update Service Account.

      Args:
        request: (AnthoseventsApiV1NamespacesServiceaccountsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/serviceaccounts/{serviceaccountsId}', http_method='PATCH', method_id='anthosevents.api.v1.namespaces.serviceaccounts.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='serviceAccount', request_type_name='AnthoseventsApiV1NamespacesServiceaccountsPatchRequest', response_type_name='ServiceAccount', supports_download=False)

    def ReplaceServiceAccount(self, request, global_params=None):
        """Rpc to replace a Service Account.

      Args:
        request: (AnthoseventsApiV1NamespacesServiceaccountsReplaceServiceAccountRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('ReplaceServiceAccount')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceServiceAccount.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/serviceaccounts/{serviceaccountsId}', http_method='PUT', method_id='anthosevents.api.v1.namespaces.serviceaccounts.replaceServiceAccount', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='serviceAccount', request_type_name='AnthoseventsApiV1NamespacesServiceaccountsReplaceServiceAccountRequest', response_type_name='ServiceAccount', supports_download=False)