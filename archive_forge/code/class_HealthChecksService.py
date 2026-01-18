from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class HealthChecksService(base_api.BaseApiService):
    """Service class for the healthChecks resource."""
    _NAME = 'healthChecks'

    def __init__(self, client):
        super(ComputeBeta.HealthChecksService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all HealthCheck resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeHealthChecksAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthChecksAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.healthChecks.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/healthChecks', request_field='', request_type_name='ComputeHealthChecksAggregatedListRequest', response_type_name='HealthChecksAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified HealthCheck resource.

      Args:
        request: (ComputeHealthChecksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.healthChecks.delete', ordered_params=['project', 'healthCheck'], path_params=['healthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/healthChecks/{healthCheck}', request_field='', request_type_name='ComputeHealthChecksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified HealthCheck resource.

      Args:
        request: (ComputeHealthChecksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheck) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.healthChecks.get', ordered_params=['project', 'healthCheck'], path_params=['healthCheck', 'project'], query_params=[], relative_path='projects/{project}/global/healthChecks/{healthCheck}', request_field='', request_type_name='ComputeHealthChecksGetRequest', response_type_name='HealthCheck', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a HealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHealthChecksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.healthChecks.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/healthChecks', request_field='healthCheck', request_type_name='ComputeHealthChecksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of HealthCheck resources available to the specified project.

      Args:
        request: (ComputeHealthChecksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HealthCheckList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.healthChecks.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/healthChecks', request_field='', request_type_name='ComputeHealthChecksListRequest', response_type_name='HealthCheckList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a HealthCheck resource in the specified project using the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeHealthChecksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.healthChecks.patch', ordered_params=['project', 'healthCheck'], path_params=['healthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/healthChecks/{healthCheck}', request_field='healthCheckResource', request_type_name='ComputeHealthChecksPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeHealthChecksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.healthChecks.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/healthChecks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeHealthChecksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a HealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHealthChecksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.healthChecks.update', ordered_params=['project', 'healthCheck'], path_params=['healthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/healthChecks/{healthCheck}', request_field='healthCheckResource', request_type_name='ComputeHealthChecksUpdateRequest', response_type_name='Operation', supports_download=False)