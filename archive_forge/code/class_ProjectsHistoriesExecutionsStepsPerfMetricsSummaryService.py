from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsPerfMetricsSummaryService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps_perfMetricsSummary resource."""
    _NAME = 'projects_histories_executions_steps_perfMetricsSummary'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsPerfMetricsSummaryService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a PerfMetricsSummary resource. Returns the existing one if it has already been created. May return any of the following error code(s): - NOT_FOUND - The containing Step does not exist.

      Args:
        request: (PerfMetricsSummary) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PerfMetricsSummary) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.steps.perfMetricsSummary.create', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfMetricsSummary', request_field='<request>', request_type_name='PerfMetricsSummary', response_type_name='PerfMetricsSummary', supports_download=False)