from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesTasksService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_tasks resource."""
    _NAME = 'projects_locations_lakes_tasks'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesTasksService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a task resource within a lake.

      Args:
        request: (DataplexProjectsLocationsLakesTasksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks', http_method='POST', method_id='dataplex.projects.locations.lakes.tasks.create', ordered_params=['parent'], path_params=['parent'], query_params=['taskId', 'validateOnly'], relative_path='v1/{+parent}/tasks', request_field='googleCloudDataplexV1Task', request_type_name='DataplexProjectsLocationsLakesTasksCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete the task resource.

      Args:
        request: (DataplexProjectsLocationsLakesTasksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}', http_method='DELETE', method_id='dataplex.projects.locations.lakes.tasks.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get task resource.

      Args:
        request: (DataplexProjectsLocationsLakesTasksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Task) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}', http_method='GET', method_id='dataplex.projects.locations.lakes.tasks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksGetRequest', response_type_name='GoogleCloudDataplexV1Task', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataplexProjectsLocationsLakesTasksGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}:getIamPolicy', http_method='GET', method_id='dataplex.projects.locations.lakes.tasks.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists tasks under the given lake.

      Args:
        request: (DataplexProjectsLocationsLakesTasksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListTasksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks', http_method='GET', method_id='dataplex.projects.locations.lakes.tasks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/tasks', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksListRequest', response_type_name='GoogleCloudDataplexV1ListTasksResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update the task resource.

      Args:
        request: (DataplexProjectsLocationsLakesTasksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}', http_method='PATCH', method_id='dataplex.projects.locations.lakes.tasks.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudDataplexV1Task', request_type_name='DataplexProjectsLocationsLakesTasksPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Run(self, request, global_params=None):
        """Run an on demand execution of a Task.

      Args:
        request: (DataplexProjectsLocationsLakesTasksRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1RunTaskResponse) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}:run', http_method='POST', method_id='dataplex.projects.locations.lakes.tasks.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='googleCloudDataplexV1RunTaskRequest', request_type_name='DataplexProjectsLocationsLakesTasksRunRequest', response_type_name='GoogleCloudDataplexV1RunTaskResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataplexProjectsLocationsLakesTasksSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}:setIamPolicy', http_method='POST', method_id='dataplex.projects.locations.lakes.tasks.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='DataplexProjectsLocationsLakesTasksSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataplexProjectsLocationsLakesTasksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}:testIamPermissions', http_method='POST', method_id='dataplex.projects.locations.lakes.tasks.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='DataplexProjectsLocationsLakesTasksTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)