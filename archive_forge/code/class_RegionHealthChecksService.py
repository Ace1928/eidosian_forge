from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionHealthChecksService(base_api.BaseApiService):
    """Service class for the regionHealthChecks resource."""
    _NAME = 'regionHealthChecks'

    def __init__(self, client):
        super(ComputeBeta.RegionHealthChecksService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified HealthCheck resource.

      Args:
        request: (ComputeRegionHealthChecksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionHealthChecks.delete', ordered_params=['project', 'region', 'healthCheck'], path_params=['healthCheck', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthChecks/{healthCheck}', request_field='', request_type_name='ComputeRegionHealthChecksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified HealthCheck resource.

      Args:
        request: (ComputeRegionHealthChecksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheck) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionHealthChecks.get', ordered_params=['project', 'region', 'healthCheck'], path_params=['healthCheck', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/healthChecks/{healthCheck}', request_field='', request_type_name='ComputeRegionHealthChecksGetRequest', response_type_name='HealthCheck', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a HealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionHealthChecksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionHealthChecks.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthChecks', request_field='healthCheck', request_type_name='ComputeRegionHealthChecksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of HealthCheck resources available to the specified project.

      Args:
        request: (ComputeRegionHealthChecksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheckList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionHealthChecks.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/healthChecks', request_field='', request_type_name='ComputeRegionHealthChecksListRequest', response_type_name='HealthCheckList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a HealthCheck resource in the specified project using the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionHealthChecksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionHealthChecks.patch', ordered_params=['project', 'region', 'healthCheck'], path_params=['healthCheck', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthChecks/{healthCheck}', request_field='healthCheckResource', request_type_name='ComputeRegionHealthChecksPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionHealthChecksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionHealthChecks.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/healthChecks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionHealthChecksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a HealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionHealthChecksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.regionHealthChecks.update', ordered_params=['project', 'region', 'healthCheck'], path_params=['healthCheck', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/healthChecks/{healthCheck}', request_field='healthCheckResource', request_type_name='ComputeRegionHealthChecksUpdateRequest', response_type_name='Operation', supports_download=False)