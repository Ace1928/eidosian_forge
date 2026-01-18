from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ApiV1NamespacesService(base_api.BaseApiService):
    """Service class for the api_v1_namespaces resource."""
    _NAME = 'api_v1_namespaces'

    def __init__(self, client):
        super(RunV1.ApiV1NamespacesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Rpc to get information about a namespace.

      Args:
        request: (RunApiV1NamespacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Namespace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}', http_method='GET', method_id='run.api.v1.namespaces.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='', request_type_name='RunApiV1NamespacesGetRequest', response_type_name='Namespace', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update a namespace.

      Args:
        request: (RunApiV1NamespacesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Namespace) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}', http_method='PATCH', method_id='run.api.v1.namespaces.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='api/v1/{+name}', request_field='namespace', request_type_name='RunApiV1NamespacesPatchRequest', response_type_name='Namespace', supports_download=False)