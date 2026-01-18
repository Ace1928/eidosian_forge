from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workflowexecutions.v1 import workflowexecutions_v1_messages as messages
class ProjectsLocationsWorkflowsExecutionsCallbacksService(base_api.BaseApiService):
    """Service class for the projects_locations_workflows_executions_callbacks resource."""
    _NAME = 'projects_locations_workflows_executions_callbacks'

    def __init__(self, client):
        super(WorkflowexecutionsV1.ProjectsLocationsWorkflowsExecutionsCallbacksService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns a list of active callbacks that belong to the execution with the given name. The returned callbacks are ordered by callback ID.

      Args:
        request: (WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCallbacksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCallbacksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workflows/{workflowsId}/executions/{executionsId}/callbacks', http_method='GET', method_id='workflowexecutions.projects.locations.workflows.executions.callbacks.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/callbacks', request_field='', request_type_name='WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCallbacksListRequest', response_type_name='ListCallbacksResponse', supports_download=False)