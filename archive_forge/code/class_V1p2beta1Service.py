from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p2beta1 import cloudasset_v1p2beta1_messages as messages
class V1p2beta1Service(base_api.BaseApiService):
    """Service class for the v1p2beta1 resource."""
    _NAME = 'v1p2beta1'

    def __init__(self, client):
        super(CloudassetV1p2beta1.V1p2beta1Service, self).__init__(client)
        self._upload_configs = {}

    def BatchGetAssetsHistory(self, request, global_params=None):
        """Batch gets the update history of assets that overlap a time window.
For RESOURCE content, this API outputs history with asset in both
non-delete or deleted status.
For IAM_POLICY content, this API outputs history when the asset and its
attached IAM POLICY both exist. This can create gaps in the output history.

      Args:
        request: (CloudassetBatchGetAssetsHistoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchGetAssetsHistoryResponse) The response message.
      """
        config = self.GetMethodConfig('BatchGetAssetsHistory')
        return self._RunMethod(config, request, global_params=global_params)
    BatchGetAssetsHistory.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1p2beta1/{v1p2beta1Id}/{v1p2beta1Id1}:batchGetAssetsHistory', http_method='GET', method_id='cloudasset.batchGetAssetsHistory', ordered_params=['parent'], path_params=['parent'], query_params=['assetNames', 'contentType', 'readTimeWindow_endTime', 'readTimeWindow_startTime'], relative_path='v1p2beta1/{+parent}:batchGetAssetsHistory', request_field='', request_type_name='CloudassetBatchGetAssetsHistoryRequest', response_type_name='BatchGetAssetsHistoryResponse', supports_download=False)

    def ExportAssets(self, request, global_params=None):
        """Exports assets with time and resource types to a given Cloud Storage.
location. The output format is newline-delimited JSON.
This API implements the google.longrunning.Operation API allowing you
to keep track of the export.

      Args:
        request: (CloudassetExportAssetsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ExportAssets')
        return self._RunMethod(config, request, global_params=global_params)
    ExportAssets.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1p2beta1/{v1p2beta1Id}/{v1p2beta1Id1}:exportAssets', http_method='POST', method_id='cloudasset.exportAssets', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1p2beta1/{+parent}:exportAssets', request_field='exportAssetsRequest', request_type_name='CloudassetExportAssetsRequest', response_type_name='Operation', supports_download=False)