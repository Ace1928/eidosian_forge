from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionTargetHttpsProxiesService(base_api.BaseApiService):
    """Service class for the regionTargetHttpsProxies resource."""
    _NAME = 'regionTargetHttpsProxies'

    def __init__(self, client):
        super(ComputeBeta.RegionTargetHttpsProxiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetHttpsProxy resource.

      Args:
        request: (ComputeRegionTargetHttpsProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionTargetHttpsProxies.delete', ordered_params=['project', 'region', 'targetHttpsProxy'], path_params=['project', 'region', 'targetHttpsProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{targetHttpsProxy}', request_field='', request_type_name='ComputeRegionTargetHttpsProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetHttpsProxy resource in the specified region.

      Args:
        request: (ComputeRegionTargetHttpsProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpsProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetHttpsProxies.get', ordered_params=['project', 'region', 'targetHttpsProxy'], path_params=['project', 'region', 'targetHttpsProxy'], query_params=[], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{targetHttpsProxy}', request_field='', request_type_name='ComputeRegionTargetHttpsProxiesGetRequest', response_type_name='TargetHttpsProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetHttpsProxy resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRegionTargetHttpsProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpsProxies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies', request_field='targetHttpsProxy', request_type_name='ComputeRegionTargetHttpsProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of TargetHttpsProxy resources available to the specified project in the specified region.

      Args:
        request: (ComputeRegionTargetHttpsProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpsProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionTargetHttpsProxies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies', request_field='', request_type_name='ComputeRegionTargetHttpsProxiesListRequest', response_type_name='TargetHttpsProxyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified regional TargetHttpsProxy resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionTargetHttpsProxiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionTargetHttpsProxies.patch', ordered_params=['project', 'region', 'targetHttpsProxy'], path_params=['project', 'region', 'targetHttpsProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{targetHttpsProxy}', request_field='targetHttpsProxyResource', request_type_name='ComputeRegionTargetHttpsProxiesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetSslCertificates(self, request, global_params=None):
        """Replaces SslCertificates for TargetHttpsProxy.

      Args:
        request: (ComputeRegionTargetHttpsProxiesSetSslCertificatesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSslCertificates')
        return self._RunMethod(config, request, global_params=global_params)
    SetSslCertificates.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpsProxies.setSslCertificates', ordered_params=['project', 'region', 'targetHttpsProxy'], path_params=['project', 'region', 'targetHttpsProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{targetHttpsProxy}/setSslCertificates', request_field='regionTargetHttpsProxiesSetSslCertificatesRequest', request_type_name='ComputeRegionTargetHttpsProxiesSetSslCertificatesRequest', response_type_name='Operation', supports_download=False)

    def SetUrlMap(self, request, global_params=None):
        """Changes the URL map for TargetHttpsProxy.

      Args:
        request: (ComputeRegionTargetHttpsProxiesSetUrlMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetUrlMap')
        return self._RunMethod(config, request, global_params=global_params)
    SetUrlMap.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpsProxies.setUrlMap', ordered_params=['project', 'region', 'targetHttpsProxy'], path_params=['project', 'region', 'targetHttpsProxy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{targetHttpsProxy}/setUrlMap', request_field='urlMapReference', request_type_name='ComputeRegionTargetHttpsProxiesSetUrlMapRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionTargetHttpsProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionTargetHttpsProxies.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/targetHttpsProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionTargetHttpsProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)