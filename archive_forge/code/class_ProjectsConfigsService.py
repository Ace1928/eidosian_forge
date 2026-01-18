from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
class ProjectsConfigsService(base_api.BaseApiService):
    """Service class for the projects_configs resource."""
    _NAME = 'projects_configs'

    def __init__(self, client):
        super(RuntimeconfigV1beta1.ProjectsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new RuntimeConfig resource. The configuration name must be unique within project.

      Args:
        request: (RuntimeconfigProjectsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RuntimeConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs', http_method='POST', method_id='runtimeconfig.projects.configs.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1beta1/{+parent}/configs', request_field='runtimeConfig', request_type_name='RuntimeconfigProjectsConfigsCreateRequest', response_type_name='RuntimeConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a RuntimeConfig resource.

      Args:
        request: (RuntimeconfigProjectsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}', http_method='DELETE', method_id='runtimeconfig.projects.configs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a RuntimeConfig resource.

      Args:
        request: (RuntimeconfigProjectsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RuntimeConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}', http_method='GET', method_id='runtimeconfig.projects.configs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsGetRequest', response_type_name='RuntimeConfig', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (RuntimeconfigProjectsConfigsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}:getIamPolicy', http_method='GET', method_id='runtimeconfig.projects.configs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta1/{+resource}:getIamPolicy', request_field='', request_type_name='RuntimeconfigProjectsConfigsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the RuntimeConfig resources within project.

      Args:
        request: (RuntimeconfigProjectsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs', http_method='GET', method_id='runtimeconfig.projects.configs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/configs', request_field='', request_type_name='RuntimeconfigProjectsConfigsListRequest', response_type_name='ListConfigsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (RuntimeconfigProjectsConfigsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}:setIamPolicy', http_method='POST', method_id='runtimeconfig.projects.configs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='RuntimeconfigProjectsConfigsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (RuntimeconfigProjectsConfigsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}:testIamPermissions', http_method='POST', method_id='runtimeconfig.projects.configs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='RuntimeconfigProjectsConfigsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a RuntimeConfig resource. The configuration must exist beforehand.

      Args:
        request: (RuntimeConfig) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RuntimeConfig) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}', http_method='PUT', method_id='runtimeconfig.projects.configs.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='<request>', request_type_name='RuntimeConfig', response_type_name='RuntimeConfig', supports_download=False)