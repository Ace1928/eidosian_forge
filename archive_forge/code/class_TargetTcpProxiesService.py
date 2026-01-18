from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetTcpProxiesService(base_api.BaseApiService):
    """Service class for the targetTcpProxies resource."""
    _NAME = 'targetTcpProxies'

    def __init__(self, client):
        super(ComputeBeta.TargetTcpProxiesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all TargetTcpProxy resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeTargetTcpProxiesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetTcpProxyAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetTcpProxies.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/targetTcpProxies', request_field='', request_type_name='ComputeTargetTcpProxiesAggregatedListRequest', response_type_name='TargetTcpProxyAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetTcpProxy resource.

      Args:
        request: (ComputeTargetTcpProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetTcpProxies.delete', ordered_params=['project', 'targetTcpProxy'], path_params=['project', 'targetTcpProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetTcpProxies/{targetTcpProxy}', request_field='', request_type_name='ComputeTargetTcpProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetTcpProxy resource.

      Args:
        request: (ComputeTargetTcpProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetTcpProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetTcpProxies.get', ordered_params=['project', 'targetTcpProxy'], path_params=['project', 'targetTcpProxy'], query_params=[], relative_path='projects/{project}/global/targetTcpProxies/{targetTcpProxy}', request_field='', request_type_name='ComputeTargetTcpProxiesGetRequest', response_type_name='TargetTcpProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetTcpProxy resource in the specified project using the data included in the request.

      Args:
        request: (ComputeTargetTcpProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetTcpProxies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/targetTcpProxies', request_field='targetTcpProxy', request_type_name='ComputeTargetTcpProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of TargetTcpProxy resources available to the specified project.

      Args:
        request: (ComputeTargetTcpProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetTcpProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetTcpProxies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/targetTcpProxies', request_field='', request_type_name='ComputeTargetTcpProxiesListRequest', response_type_name='TargetTcpProxyList', supports_download=False)

    def SetBackendService(self, request, global_params=None):
        """Changes the BackendService for TargetTcpProxy.

      Args:
        request: (ComputeTargetTcpProxiesSetBackendServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetBackendService')
        return self._RunMethod(config, request, global_params=global_params)
    SetBackendService.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetTcpProxies.setBackendService', ordered_params=['project', 'targetTcpProxy'], path_params=['project', 'targetTcpProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetTcpProxies/{targetTcpProxy}/setBackendService', request_field='targetTcpProxiesSetBackendServiceRequest', request_type_name='ComputeTargetTcpProxiesSetBackendServiceRequest', response_type_name='Operation', supports_download=False)

    def SetProxyHeader(self, request, global_params=None):
        """Changes the ProxyHeaderType for TargetTcpProxy.

      Args:
        request: (ComputeTargetTcpProxiesSetProxyHeaderRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetProxyHeader')
        return self._RunMethod(config, request, global_params=global_params)
    SetProxyHeader.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetTcpProxies.setProxyHeader', ordered_params=['project', 'targetTcpProxy'], path_params=['project', 'targetTcpProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetTcpProxies/{targetTcpProxy}/setProxyHeader', request_field='targetTcpProxiesSetProxyHeaderRequest', request_type_name='ComputeTargetTcpProxiesSetProxyHeaderRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetTcpProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetTcpProxies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/targetTcpProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetTcpProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)