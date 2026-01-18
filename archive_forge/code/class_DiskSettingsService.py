from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
class DiskSettingsService(base_api.BaseApiService):
    """Service class for the diskSettings resource."""
    _NAME = 'diskSettings'

    def __init__(self, client):
        super(ComputeAlpha.DiskSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get Zonal Disk Settings.

      Args:
        request: (ComputeDiskSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.diskSettings.get', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/diskSettings', request_field='', request_type_name='ComputeDiskSettingsGetRequest', response_type_name='DiskSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch Zonal Disk Settings.

      Args:
        request: (ComputeDiskSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.diskSettings.patch', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/zones/{zone}/diskSettings', request_field='diskSettings', request_type_name='ComputeDiskSettingsPatchRequest', response_type_name='Operation', supports_download=False)