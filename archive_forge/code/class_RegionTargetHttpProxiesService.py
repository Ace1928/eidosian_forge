from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionTargetHttpProxiesService(base_api.BaseApiService):
    """Service class for the regionTargetHttpProxies resource."""
    _NAME = 'regionTargetHttpProxies'

    def __init__(self, client):
        super(ComputeBeta.RegionTargetHttpProxiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetHttpProxy resource.

      Args:
        request: (ComputeRegionTargetHttpProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionTargetHttpProxies.delete', ordered_params=['project', 'region', 'targetHttpProxy'], path_params=['project', 'region', 'targetHttpProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpProxies/{targetHttpProxy}', request_field='', request_type_name='ComputeRegionTargetHttpProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetHttpProxy resource in the specified region.

      Args:
        request: (ComputeRegionTargetHttpProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetHttpProxies.get', ordered_params=['project', 'region', 'targetHttpProxy'], path_params=['project', 'region', 'targetHttpProxy'], query_params=[], relative_path='projects/{project}/regions/{region}/targetHttpProxies/{targetHttpProxy}', request_field='', request_type_name='ComputeRegionTargetHttpProxiesGetRequest', response_type_name='TargetHttpProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetHttpProxy resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRegionTargetHttpProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpProxies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpProxies', request_field='targetHttpProxy', request_type_name='ComputeRegionTargetHttpProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of TargetHttpProxy resources available to the specified project in the specified region.

      Args:
        request: (ComputeRegionTargetHttpProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetHttpProxies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/targetHttpProxies', request_field='', request_type_name='ComputeRegionTargetHttpProxiesListRequest', response_type_name='TargetHttpProxyList', supports_download=False)

    def SetUrlMap(self, request, global_params=None):
        """Changes the URL map for TargetHttpProxy.

      Args:
        request: (ComputeRegionTargetHttpProxiesSetUrlMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetUrlMap')
        return self._RunMethod(config, request, global_params=global_params)
    SetUrlMap.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpProxies.setUrlMap', ordered_params=['project', 'region', 'targetHttpProxy'], path_params=['project', 'region', 'targetHttpProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpProxies/{targetHttpProxy}/setUrlMap', request_field='urlMapReference', request_type_name='ComputeRegionTargetHttpProxiesSetUrlMapRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionTargetHttpProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpProxies.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/targetHttpProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionTargetHttpProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)