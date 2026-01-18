from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
class RegionSnapshotSettingsService(base_api.BaseApiService):
    """Service class for the regionSnapshotSettings resource."""
    _NAME = 'regionSnapshotSettings'

    def __init__(self, client):
        super(ComputeAlpha.RegionSnapshotSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get region snapshot settings.

      Args:
        request: (ComputeRegionSnapshotSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSnapshotSettings.get', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/snapshotSettings', request_field='', request_type_name='ComputeRegionSnapshotSettingsGetRequest', response_type_name='SnapshotSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch region snapshot settings.

      Args:
        request: (ComputeRegionSnapshotSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionSnapshotSettings.patch', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/snapshotSettings', request_field='snapshotSettings', request_type_name='ComputeRegionSnapshotSettingsPatchRequest', response_type_name='Operation', supports_download=False)