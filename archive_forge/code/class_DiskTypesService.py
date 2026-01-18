from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class DiskTypesService(base_api.BaseApiService):
    """Service class for the diskTypes resource."""
    _NAME = 'diskTypes'

    def __init__(self, client):
        super(ComputeBeta.DiskTypesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of disk types. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeDiskTypesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskTypeAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.diskTypes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/diskTypes', request_field='', request_type_name='ComputeDiskTypesAggregatedListRequest', response_type_name='DiskTypeAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified disk type.

      Args:
        request: (ComputeDiskTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.diskTypes.get', ordered_params=['project', 'zone', 'diskType'], path_params=['diskType', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/diskTypes/{diskType}', request_field='', request_type_name='ComputeDiskTypesGetRequest', response_type_name='DiskType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of disk types available to the specified project.

      Args:
        request: (ComputeDiskTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.diskTypes.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/diskTypes', request_field='', request_type_name='ComputeDiskTypesListRequest', response_type_name='DiskTypeList', supports_download=False)