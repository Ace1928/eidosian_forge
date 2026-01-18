from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionHealthCheckServicesService(base_api.BaseApiService):
    """Service class for the regionHealthCheckServices resource."""
    _NAME = 'regionHealthCheckServices'

    def __init__(self, client):
        super(ComputeBeta.RegionHealthCheckServicesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified regional HealthCheckService.

      Args:
        request: (ComputeRegionHealthCheckServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionHealthCheckServices.delete', ordered_params=['project', 'region', 'healthCheckService'], path_params=['healthCheckService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthCheckServices/{healthCheckService}', request_field='', request_type_name='ComputeRegionHealthCheckServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified regional HealthCheckService resource.

      Args:
        request: (ComputeRegionHealthCheckServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheckService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionHealthCheckServices.get', ordered_params=['project', 'region', 'healthCheckService'], path_params=['healthCheckService', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/healthCheckServices/{healthCheckService}', request_field='', request_type_name='ComputeRegionHealthCheckServicesGetRequest', response_type_name='HealthCheckService', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a regional HealthCheckService resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRegionHealthCheckServicesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionHealthCheckServices.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthCheckServices', request_field='healthCheckService', request_type_name='ComputeRegionHealthCheckServicesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the HealthCheckService resources that have been configured for the specified project in the given region.

      Args:
        request: (ComputeRegionHealthCheckServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheckServicesList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionHealthCheckServices.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/healthCheckServices', request_field='', request_type_name='ComputeRegionHealthCheckServicesListRequest', response_type_name='HealthCheckServicesList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified regional HealthCheckService resource with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionHealthCheckServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionHealthCheckServices.patch', ordered_params=['project', 'region', 'healthCheckService'], path_params=['healthCheckService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthCheckServices/{healthCheckService}', request_field='healthCheckServiceResource', request_type_name='ComputeRegionHealthCheckServicesPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionHealthCheckServicesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionHealthCheckServices.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/healthCheckServices/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionHealthCheckServicesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)