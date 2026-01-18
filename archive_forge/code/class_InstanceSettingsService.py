from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InstanceSettingsService(base_api.BaseApiService):
    """Service class for the instanceSettings resource."""
    _NAME = 'instanceSettings'

    def __init__(self, client):
        super(ComputeBeta.InstanceSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get Instance settings.

      Args:
        request: (ComputeInstanceSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceSettings.get', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceSettings', request_field='', request_type_name='ComputeInstanceSettingsGetRequest', response_type_name='InstanceSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch Instance settings.

      Args:
        request: (ComputeInstanceSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.instanceSettings.patch', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/zones/{zone}/instanceSettings', request_field='instanceSettings', request_type_name='ComputeInstanceSettingsPatchRequest', response_type_name='Operation', supports_download=False)