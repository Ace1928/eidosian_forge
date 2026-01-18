from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsWorkflowTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_workflowTemplates resource."""
    _NAME = 'projects_locations_workflowTemplates'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsWorkflowTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates new workflow template.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkflowTemplate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/workflowTemplates', request_field='workflowTemplate', request_type_name='DataprocProjectsLocationsWorkflowTemplatesCreateRequest', response_type_name='WorkflowTemplate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a workflow template. It does not cancel in-progress workflows.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}', http_method='DELETE', method_id='dataproc.projects.locations.workflowTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=['version'], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsWorkflowTemplatesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the latest workflow template.Can retrieve previously instantiated template by specifying optional version parameter.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkflowTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}', http_method='GET', method_id='dataproc.projects.locations.workflowTemplates.get', ordered_params=['name'], path_params=['name'], query_params=['version'], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsWorkflowTemplatesGetRequest', response_type_name='WorkflowTemplate', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}:getIamPolicy', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DataprocProjectsLocationsWorkflowTemplatesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Instantiate(self, request, global_params=None):
        """Instantiates a template and begins execution.The returned Operation can be used to track execution of workflow by polling operations.get. The Operation will complete when entire workflow is finished.The running workflow can be aborted via operations.cancel. This will cause any inflight jobs to be cancelled and workflow-owned clusters to be deleted.The Operation.metadata will be WorkflowMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#workflowmetadata). Also see Using WorkflowMetadata (https://cloud.google.com/dataproc/docs/concepts/workflows/debugging#using_workflowmetadata).On successful completion, Operation.response will be Empty.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesInstantiateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Instantiate')
        return self._RunMethod(config, request, global_params=global_params)
    Instantiate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}:instantiate', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.instantiate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:instantiate', request_field='instantiateWorkflowTemplateRequest', request_type_name='DataprocProjectsLocationsWorkflowTemplatesInstantiateRequest', response_type_name='Operation', supports_download=False)

    def InstantiateInline(self, request, global_params=None):
        """Instantiates a template and begins execution.This method is equivalent to executing the sequence CreateWorkflowTemplate, InstantiateWorkflowTemplate, DeleteWorkflowTemplate.The returned Operation can be used to track execution of workflow by polling operations.get. The Operation will complete when entire workflow is finished.The running workflow can be aborted via operations.cancel. This will cause any inflight jobs to be cancelled and workflow-owned clusters to be deleted.The Operation.metadata will be WorkflowMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#workflowmetadata). Also see Using WorkflowMetadata (https://cloud.google.com/dataproc/docs/concepts/workflows/debugging#using_workflowmetadata).On successful completion, Operation.response will be Empty.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesInstantiateInlineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InstantiateInline')
        return self._RunMethod(config, request, global_params=global_params)
    InstantiateInline.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates:instantiateInline', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.instantiateInline', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1/{+parent}/workflowTemplates:instantiateInline', request_field='workflowTemplate', request_type_name='DataprocProjectsLocationsWorkflowTemplatesInstantiateInlineRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists workflows that match the specified filter in the request.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkflowTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates', http_method='GET', method_id='dataproc.projects.locations.workflowTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/workflowTemplates', request_field='', request_type_name='DataprocProjectsLocationsWorkflowTemplatesListRequest', response_type_name='ListWorkflowTemplatesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}:setIamPolicy', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DataprocProjectsLocationsWorkflowTemplatesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataprocProjectsLocationsWorkflowTemplatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}:testIamPermissions', http_method='POST', method_id='dataproc.projects.locations.workflowTemplates.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DataprocProjectsLocationsWorkflowTemplatesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates (replaces) workflow template. The updated template must contain version that matches the current server version.

      Args:
        request: (WorkflowTemplate) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkflowTemplate) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflowTemplates/{workflowTemplatesId}', http_method='PUT', method_id='dataproc.projects.locations.workflowTemplates.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='WorkflowTemplate', response_type_name='WorkflowTemplate', supports_download=False)