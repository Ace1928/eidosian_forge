from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ApiV1NamespacesSecretsService(base_api.BaseApiService):
    """Service class for the api_v1_namespaces_secrets resource."""
    _NAME = 'api_v1_namespaces_secrets'

    def __init__(self, client):
        super(RunV1.ApiV1NamespacesSecretsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new secret.

      Args:
        request: (RunApiV1NamespacesSecretsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/secrets', http_method='POST', method_id='run.api.v1.namespaces.secrets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='api/v1/{+parent}/secrets', request_field='secret', request_type_name='RunApiV1NamespacesSecretsCreateRequest', response_type_name='Secret', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a secret.

      Args:
        request: (RunApiV1NamespacesSecretsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/secrets/{secretsId}', http_method='GET', method_id='run.api.v1.namespaces.secrets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='', request_type_name='RunApiV1NamespacesSecretsGetRequest', response_type_name='Secret', supports_download=False)

    def ReplaceSecret(self, request, global_params=None):
        """Rpc to replace a secret. Only the spec, metadata labels, and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (RunApiV1NamespacesSecretsReplaceSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('ReplaceSecret')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceSecret.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/secrets/{secretsId}', http_method='PUT', method_id='run.api.v1.namespaces.secrets.replaceSecret', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='secret', request_type_name='RunApiV1NamespacesSecretsReplaceSecretRequest', response_type_name='Secret', supports_download=False)