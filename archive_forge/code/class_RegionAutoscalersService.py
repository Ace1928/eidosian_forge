from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionAutoscalersService(base_api.BaseApiService):
    """Service class for the regionAutoscalers resource."""
    _NAME = 'regionAutoscalers'

    def __init__(self, client):
        super(ComputeBeta.RegionAutoscalersService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified autoscaler.

      Args:
        request: (ComputeRegionAutoscalersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionAutoscalers.delete', ordered_params=['project', 'region', 'autoscaler'], path_params=['autoscaler', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/autoscalers/{autoscaler}', request_field='', request_type_name='ComputeRegionAutoscalersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified autoscaler.

      Args:
        request: (ComputeRegionAutoscalersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Autoscaler) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionAutoscalers.get', ordered_params=['project', 'region', 'autoscaler'], path_params=['autoscaler', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/autoscalers/{autoscaler}', request_field='', request_type_name='ComputeRegionAutoscalersGetRequest', response_type_name='Autoscaler', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an autoscaler in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionAutoscalersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionAutoscalers.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/autoscalers', request_field='autoscaler', request_type_name='ComputeRegionAutoscalersInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of autoscalers contained within the specified region.

      Args:
        request: (ComputeRegionAutoscalersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionAutoscalerList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionAutoscalers.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/autoscalers', request_field='', request_type_name='ComputeRegionAutoscalersListRequest', response_type_name='RegionAutoscalerList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an autoscaler in the specified project using the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionAutoscalersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionAutoscalers.patch', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['autoscaler', 'requestId'], relative_path='projects/{project}/regions/{region}/autoscalers', request_field='autoscalerResource', request_type_name='ComputeRegionAutoscalersPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionAutoscalersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionAutoscalers.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/autoscalers/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionAutoscalersTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an autoscaler in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionAutoscalersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.regionAutoscalers.update', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['autoscaler', 'requestId'], relative_path='projects/{project}/regions/{region}/autoscalers', request_field='autoscalerResource', request_type_name='ComputeRegionAutoscalersUpdateRequest', response_type_name='Operation', supports_download=False)