from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionUrlMapsService(base_api.BaseApiService):
    """Service class for the regionUrlMaps resource."""
    _NAME = 'regionUrlMaps'

    def __init__(self, client):
        super(ComputeBeta.RegionUrlMapsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified UrlMap resource.

      Args:
        request: (ComputeRegionUrlMapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionUrlMaps.delete', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}', request_field='', request_type_name='ComputeRegionUrlMapsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified UrlMap resource.

      Args:
        request: (ComputeRegionUrlMapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionUrlMaps.get', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=[], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}', request_field='', request_type_name='ComputeRegionUrlMapsGetRequest', response_type_name='UrlMap', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a UrlMap resource in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionUrlMapsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionUrlMaps.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/urlMaps', request_field='urlMap', request_type_name='ComputeRegionUrlMapsInsertRequest', response_type_name='Operation', supports_download=False)

    def InvalidateCache(self, request, global_params=None):
        """Initiates a cache invalidation operation, invalidating the specified path, scoped to the specified UrlMap. For more information, see [Invalidating cached content](/cdn/docs/invalidating-cached-content).

      Args:
        request: (ComputeRegionUrlMapsInvalidateCacheRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InvalidateCache')
        return self._RunMethod(config, request, global_params=global_params)
    InvalidateCache.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionUrlMaps.invalidateCache', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}/invalidateCache', request_field='cacheInvalidationRule', request_type_name='ComputeRegionUrlMapsInvalidateCacheRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of UrlMap resources available to the specified project in the specified region.

      Args:
        request: (ComputeRegionUrlMapsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMapList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionUrlMaps.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/urlMaps', request_field='', request_type_name='ComputeRegionUrlMapsListRequest', response_type_name='UrlMapList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified UrlMap resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionUrlMapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionUrlMaps.patch', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}', request_field='urlMapResource', request_type_name='ComputeRegionUrlMapsPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionUrlMapsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionUrlMaps.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/urlMaps/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionUrlMapsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified UrlMap resource with the data included in the request.

      Args:
        request: (ComputeRegionUrlMapsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.regionUrlMaps.update', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}', request_field='urlMapResource', request_type_name='ComputeRegionUrlMapsUpdateRequest', response_type_name='Operation', supports_download=False)

    def Validate(self, request, global_params=None):
        """Runs static validation for the UrlMap. In particular, the tests of the provided UrlMap will be run. Calling this method does NOT create the UrlMap.

      Args:
        request: (ComputeRegionUrlMapsValidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMapsValidateResponse) The response message.
      """
        config = self.GetMethodConfig('Validate')
        return self._RunMethod(config, request, global_params=global_params)
    Validate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionUrlMaps.validate', ordered_params=['project', 'region', 'urlMap'], path_params=['project', 'region', 'urlMap'], query_params=[], relative_path='projects/{project}/regions/{region}/urlMaps/{urlMap}/validate', request_field='regionUrlMapsValidateRequest', request_type_name='ComputeRegionUrlMapsValidateRequest', response_type_name='UrlMapsValidateResponse', supports_download=False)