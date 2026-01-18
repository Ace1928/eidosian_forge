from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionTargetTcpProxiesService(base_api.BaseApiService):
    """Service class for the regionTargetTcpProxies resource."""
    _NAME = 'regionTargetTcpProxies'

    def __init__(self, client):
        super(ComputeBeta.RegionTargetTcpProxiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetTcpProxy resource.

      Args:
        request: (ComputeRegionTargetTcpProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionTargetTcpProxies.delete', ordered_params=['project', 'region', 'targetTcpProxy'], path_params=['project', 'region', 'targetTcpProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetTcpProxies/{targetTcpProxy}', request_field='', request_type_name='ComputeRegionTargetTcpProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetTcpProxy resource.

      Args:
        request: (ComputeRegionTargetTcpProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetTcpProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetTcpProxies.get', ordered_params=['project', 'region', 'targetTcpProxy'], path_params=['project', 'region', 'targetTcpProxy'], query_params=[], relative_path='projects/{project}/regions/{region}/targetTcpProxies/{targetTcpProxy}', request_field='', request_type_name='ComputeRegionTargetTcpProxiesGetRequest', response_type_name='TargetTcpProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetTcpProxy resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRegionTargetTcpProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetTcpProxies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetTcpProxies', request_field='targetTcpProxy', request_type_name='ComputeRegionTargetTcpProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of TargetTcpProxy resources available to the specified project in a given region.

      Args:
        request: (ComputeRegionTargetTcpProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetTcpProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetTcpProxies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/targetTcpProxies', request_field='', request_type_name='ComputeRegionTargetTcpProxiesListRequest', response_type_name='TargetTcpProxyList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionTargetTcpProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetTcpProxies.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/targetTcpProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionTargetTcpProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)