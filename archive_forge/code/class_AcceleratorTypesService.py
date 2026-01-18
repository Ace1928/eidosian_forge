from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class AcceleratorTypesService(base_api.BaseApiService):
    """Service class for the acceleratorTypes resource."""
    _NAME = 'acceleratorTypes'

    def __init__(self, client):
        super(ComputeBeta.AcceleratorTypesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of accelerator types. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeAcceleratorTypesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AcceleratorTypeAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.acceleratorTypes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/acceleratorTypes', request_field='', request_type_name='ComputeAcceleratorTypesAggregatedListRequest', response_type_name='AcceleratorTypeAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified accelerator type.

      Args:
        request: (ComputeAcceleratorTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AcceleratorType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.acceleratorTypes.get', ordered_params=['project', 'zone', 'acceleratorType'], path_params=['acceleratorType', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/acceleratorTypes/{acceleratorType}', request_field='', request_type_name='ComputeAcceleratorTypesGetRequest', response_type_name='AcceleratorType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of accelerator types that are available to the specified project.

      Args:
        request: (ComputeAcceleratorTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AcceleratorTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.acceleratorTypes.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/acceleratorTypes', request_field='', request_type_name='ComputeAcceleratorTypesListRequest', response_type_name='AcceleratorTypeList', supports_download=False)