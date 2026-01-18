from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class StoragePoolTypesService(base_api.BaseApiService):
    """Service class for the storagePoolTypes resource."""
    _NAME = 'storagePoolTypes'

    def __init__(self, client):
        super(ComputeBeta.StoragePoolTypesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of storage pool types. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeStoragePoolTypesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolTypeAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePoolTypes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/storagePoolTypes', request_field='', request_type_name='ComputeStoragePoolTypesAggregatedListRequest', response_type_name='StoragePoolTypeAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified storage pool type.

      Args:
        request: (ComputeStoragePoolTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePoolTypes.get', ordered_params=['project', 'zone', 'storagePoolType'], path_params=['project', 'storagePoolType', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/storagePoolTypes/{storagePoolType}', request_field='', request_type_name='ComputeStoragePoolTypesGetRequest', response_type_name='StoragePoolType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of storage pool types available to the specified project.

      Args:
        request: (ComputeStoragePoolTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePoolTypes.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/storagePoolTypes', request_field='', request_type_name='ComputeStoragePoolTypesListRequest', response_type_name='StoragePoolTypeList', supports_download=False)