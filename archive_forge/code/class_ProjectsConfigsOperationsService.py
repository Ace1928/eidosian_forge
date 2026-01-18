from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
class ProjectsConfigsOperationsService(base_api.BaseApiService):
    """Service class for the projects_configs_operations resource."""
    _NAME = 'projects_configs_operations'

    def __init__(self, client):
        super(RuntimeconfigV1beta1.ProjectsConfigsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (RuntimeconfigProjectsConfigsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/operations/{operationsId}', http_method='GET', method_id='runtimeconfig.projects.configs.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (RuntimeconfigProjectsConfigsOperationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/operations/{operationsId}:testIamPermissions', http_method='POST', method_id='runtimeconfig.projects.configs.operations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='RuntimeconfigProjectsConfigsOperationsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)