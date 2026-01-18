from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NetworkEdgeSecurityServicesService(base_api.BaseApiService):
    """Service class for the networkEdgeSecurityServices resource."""
    _NAME = 'networkEdgeSecurityServices'

    def __init__(self, client):
        super(ComputeBeta.NetworkEdgeSecurityServicesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all NetworkEdgeSecurityService resources available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNetworkEdgeSecurityServicesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEdgeSecurityServiceAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkEdgeSecurityServices.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/networkEdgeSecurityServices', request_field='', request_type_name='ComputeNetworkEdgeSecurityServicesAggregatedListRequest', response_type_name='NetworkEdgeSecurityServiceAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified service.

      Args:
        request: (ComputeNetworkEdgeSecurityServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.networkEdgeSecurityServices.delete', ordered_params=['project', 'region', 'networkEdgeSecurityService'], path_params=['networkEdgeSecurityService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkEdgeSecurityServices/{networkEdgeSecurityService}', request_field='', request_type_name='ComputeNetworkEdgeSecurityServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a specified NetworkEdgeSecurityService.

      Args:
        request: (ComputeNetworkEdgeSecurityServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEdgeSecurityService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkEdgeSecurityServices.get', ordered_params=['project', 'region', 'networkEdgeSecurityService'], path_params=['networkEdgeSecurityService', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/networkEdgeSecurityServices/{networkEdgeSecurityService}', request_field='', request_type_name='ComputeNetworkEdgeSecurityServicesGetRequest', response_type_name='NetworkEdgeSecurityService', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new service in the specified project using the data included in the request.

      Args:
        request: (ComputeNetworkEdgeSecurityServicesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEdgeSecurityServices.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId', 'validateOnly'], relative_path='projects/{project}/regions/{region}/networkEdgeSecurityServices', request_field='networkEdgeSecurityService', request_type_name='ComputeNetworkEdgeSecurityServicesInsertRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified policy with the data included in the request.

      Args:
        request: (ComputeNetworkEdgeSecurityServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.networkEdgeSecurityServices.patch', ordered_params=['project', 'region', 'networkEdgeSecurityService'], path_params=['networkEdgeSecurityService', 'project', 'region'], query_params=['paths', 'requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/networkEdgeSecurityServices/{networkEdgeSecurityService}', request_field='networkEdgeSecurityServiceResource', request_type_name='ComputeNetworkEdgeSecurityServicesPatchRequest', response_type_name='Operation', supports_download=False)