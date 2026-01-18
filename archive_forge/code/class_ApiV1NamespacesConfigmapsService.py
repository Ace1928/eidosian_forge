from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ApiV1NamespacesConfigmapsService(base_api.BaseApiService):
    """Service class for the api_v1_namespaces_configmaps resource."""
    _NAME = 'api_v1_namespaces_configmaps'

    def __init__(self, client):
        super(AnthoseventsV1.ApiV1NamespacesConfigmapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new config map.

      Args:
        request: (AnthoseventsApiV1NamespacesConfigmapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/configmaps', http_method='POST', method_id='anthosevents.api.v1.namespaces.configmaps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='api/v1/{+parent}/configmaps', request_field='configMap', request_type_name='AnthoseventsApiV1NamespacesConfigmapsCreateRequest', response_type_name='ConfigMap', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to retrieve config map.

      Args:
        request: (AnthoseventsApiV1NamespacesConfigmapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/configmaps/{configmapsId}', http_method='GET', method_id='anthosevents.api.v1.namespaces.configmaps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='', request_type_name='AnthoseventsApiV1NamespacesConfigmapsGetRequest', response_type_name='ConfigMap', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update a ConfigMap.

      Args:
        request: (AnthoseventsApiV1NamespacesConfigmapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/configmaps/{configmapsId}', http_method='PATCH', method_id='anthosevents.api.v1.namespaces.configmaps.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='configMap', request_type_name='AnthoseventsApiV1NamespacesConfigmapsPatchRequest', response_type_name='ConfigMap', supports_download=False)

    def ReplaceConfigMap(self, request, global_params=None):
        """Rpc to replace a ConfigMap.

      Args:
        request: (AnthoseventsApiV1NamespacesConfigmapsReplaceConfigMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('ReplaceConfigMap')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceConfigMap.method_config = lambda: base_api.ApiMethodInfo(flat_path='api/v1/namespaces/{namespacesId}/configmaps/{configmapsId}', http_method='PUT', method_id='anthosevents.api.v1.namespaces.configmaps.replaceConfigMap', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='api/v1/{+name}', request_field='configMap', request_type_name='AnthoseventsApiV1NamespacesConfigmapsReplaceConfigMapRequest', response_type_name='ConfigMap', supports_download=False)