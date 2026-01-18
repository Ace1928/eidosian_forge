from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composerflex.v1alpha1 import composerflex_v1alpha1_messages as messages
class ProjectsLocationsWorkflowsRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_workflows_runs resource."""
    _NAME = 'projects_locations_workflows_runs'

    def __init__(self, client):
        super(ComposerflexV1alpha1.ProjectsLocationsWorkflowsRunsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a workflow run.

      Args:
        request: (ComposerflexProjectsLocationsWorkflowsRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkflowRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workflows/{workflowsId}/runs/{runsId}', http_method='GET', method_id='composerflex.projects.locations.workflows.runs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ComposerflexProjectsLocationsWorkflowsRunsGetRequest', response_type_name='WorkflowRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists runs of a workflow in a project and location. If the workflow is set to the wildcard "-", then workflow runs from all workflows in the project and location will be listed.

      Args:
        request: (ComposerflexProjectsLocationsWorkflowsRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkflowRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workflows/{workflowsId}/runs', http_method='GET', method_id='composerflex.projects.locations.workflows.runs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/runs', request_field='', request_type_name='ComposerflexProjectsLocationsWorkflowsRunsListRequest', response_type_name='ListWorkflowRunsResponse', supports_download=False)