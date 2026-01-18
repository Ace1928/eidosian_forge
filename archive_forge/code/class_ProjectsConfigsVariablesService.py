from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
class ProjectsConfigsVariablesService(base_api.BaseApiService):
    """Service class for the projects_configs_variables resource."""
    _NAME = 'projects_configs_variables'

    def __init__(self, client):
        super(RuntimeconfigV1beta1.ProjectsConfigsVariablesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a variable within the given configuration. You cannot create a variable with a name that is a prefix of an existing variable name, or a name that has an existing variable name as a prefix. To learn more about creating a variable, read the [Setting and Getting Data](/deployment-manager/runtime-configurator/set-and-get-variables) documentation.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Variable) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables', http_method='POST', method_id='runtimeconfig.projects.configs.variables.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1beta1/{+parent}/variables', request_field='variable', request_type_name='RuntimeconfigProjectsConfigsVariablesCreateRequest', response_type_name='Variable', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a variable or multiple variables. If you specify a variable name, then that variable is deleted. If you specify a prefix and `recursive` is true, then all variables with that prefix are deleted. You must set a `recursive` to true if you delete variables by prefix.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables/{variablesId}', http_method='DELETE', method_id='runtimeconfig.projects.configs.variables.delete', ordered_params=['name'], path_params=['name'], query_params=['recursive'], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsVariablesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a single variable.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Variable) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables/{variablesId}', http_method='GET', method_id='runtimeconfig.projects.configs.variables.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsVariablesGetRequest', response_type_name='Variable', supports_download=False)

    def List(self, request, global_params=None):
        """Lists variables within given a configuration, matching any provided filters. This only lists variable names, not the values, unless `return_values` is true, in which case only variables that user has IAM permission to GetVariable will be returned.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVariablesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables', http_method='GET', method_id='runtimeconfig.projects.configs.variables.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'returnValues'], relative_path='v1beta1/{+parent}/variables', request_field='', request_type_name='RuntimeconfigProjectsConfigsVariablesListRequest', response_type_name='ListVariablesResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables/{variablesId}:testIamPermissions', http_method='POST', method_id='runtimeconfig.projects.configs.variables.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='RuntimeconfigProjectsConfigsVariablesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing variable with a new value.

      Args:
        request: (Variable) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Variable) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables/{variablesId}', http_method='PUT', method_id='runtimeconfig.projects.configs.variables.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='<request>', request_type_name='Variable', response_type_name='Variable', supports_download=False)

    def Watch(self, request, global_params=None):
        """Watches a specific variable and waits for a change in the variable's value. When there is a change, this method returns the new value or times out. If a variable is deleted while being watched, the `variableState` state is set to `DELETED` and the method returns the last known variable `value`. If you set the deadline for watching to a larger value than internal timeout (60 seconds), the current variable value is returned and the `variableState` will be `VARIABLE_STATE_UNSPECIFIED`. To learn more about creating a watcher, read the [Watching a Variable for Changes](/deployment-manager/runtime-configurator/watching-a-variable) documentation.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesWatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Variable) The response message.
      """
        config = self.GetMethodConfig('Watch')
        return self._RunMethod(config, request, global_params=global_params)
    Watch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/variables/{variablesId}:watch', http_method='POST', method_id='runtimeconfig.projects.configs.variables.watch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:watch', request_field='watchVariableRequest', request_type_name='RuntimeconfigProjectsConfigsVariablesWatchRequest', response_type_name='Variable', supports_download=False)