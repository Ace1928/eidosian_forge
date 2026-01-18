from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class HttpsHealthChecksService(base_api.BaseApiService):
    """Service class for the httpsHealthChecks resource."""
    _NAME = 'httpsHealthChecks'

    def __init__(self, client):
        super(ComputeBeta.HttpsHealthChecksService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified HttpsHealthCheck resource.

      Args:
        request: (ComputeHttpsHealthChecksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.httpsHealthChecks.delete', ordered_params=['project', 'httpsHealthCheck'], path_params=['httpsHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpsHealthChecks/{httpsHealthCheck}', request_field='', request_type_name='ComputeHttpsHealthChecksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified HttpsHealthCheck resource.

      Args:
        request: (ComputeHttpsHealthChecksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpsHealthCheck) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.httpsHealthChecks.get', ordered_params=['project', 'httpsHealthCheck'], path_params=['httpsHealthCheck', 'project'], query_params=[], relative_path='projects/{project}/global/httpsHealthChecks/{httpsHealthCheck}', request_field='', request_type_name='ComputeHttpsHealthChecksGetRequest', response_type_name='HttpsHealthCheck', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a HttpsHealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHttpsHealthChecksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.httpsHealthChecks.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/httpsHealthChecks', request_field='httpsHealthCheck', request_type_name='ComputeHttpsHealthChecksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of HttpsHealthCheck resources available to the specified project.

      Args:
        request: (ComputeHttpsHealthChecksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpsHealthCheckList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.httpsHealthChecks.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/httpsHealthChecks', request_field='', request_type_name='ComputeHttpsHealthChecksListRequest', response_type_name='HttpsHealthCheckList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a HttpsHealthCheck resource in the specified project using the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeHttpsHealthChecksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.httpsHealthChecks.patch', ordered_params=['project', 'httpsHealthCheck'], path_params=['httpsHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpsHealthChecks/{httpsHealthCheck}', request_field='httpsHealthCheckResource', request_type_name='ComputeHttpsHealthChecksPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeHttpsHealthChecksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.httpsHealthChecks.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/httpsHealthChecks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeHttpsHealthChecksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a HttpsHealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHttpsHealthChecksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.httpsHealthChecks.update', ordered_params=['project', 'httpsHealthCheck'], path_params=['httpsHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpsHealthChecks/{httpsHealthCheck}', request_field='httpsHealthCheckResource', request_type_name='ComputeHttpsHealthChecksUpdateRequest', response_type_name='Operation', supports_download=False)