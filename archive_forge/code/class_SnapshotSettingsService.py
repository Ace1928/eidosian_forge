from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SnapshotSettingsService(base_api.BaseApiService):
    """Service class for the snapshotSettings resource."""
    _NAME = 'snapshotSettings'

    def __init__(self, client):
        super(ComputeBeta.SnapshotSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get snapshot settings.

      Args:
        request: (ComputeSnapshotSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.snapshotSettings.get', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='projects/{project}/global/snapshotSettings', request_field='', request_type_name='ComputeSnapshotSettingsGetRequest', response_type_name='SnapshotSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch snapshot settings.

      Args:
        request: (ComputeSnapshotSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.snapshotSettings.patch', ordered_params=['project'], path_params=['project'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/global/snapshotSettings', request_field='snapshotSettings', request_type_name='ComputeSnapshotSettingsPatchRequest', response_type_name='Operation', supports_download=False)