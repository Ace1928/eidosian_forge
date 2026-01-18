from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsPerfSampleSeriesService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps_perfSampleSeries resource."""
    _NAME = 'projects_histories_executions_steps_perfSampleSeries'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsPerfSampleSeriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a PerfSampleSeries. May return any of the following error code(s): - ALREADY_EXISTS - PerfMetricSummary already exists for the given Step - NOT_FOUND - The containing Step does not exist.

      Args:
        request: (PerfSampleSeries) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PerfSampleSeries) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.steps.perfSampleSeries.create', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfSampleSeries', request_field='<request>', request_type_name='PerfSampleSeries', response_type_name='PerfSampleSeries', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a PerfSampleSeries. May return any of the following error code(s): - NOT_FOUND - The specified PerfSampleSeries does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PerfSampleSeries) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.perfSampleSeries.get', ordered_params=['projectId', 'historyId', 'executionId', 'stepId', 'sampleSeriesId'], path_params=['executionId', 'historyId', 'projectId', 'sampleSeriesId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfSampleSeries/{sampleSeriesId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesGetRequest', response_type_name='PerfSampleSeries', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PerfSampleSeries for a given Step. The request provides an optional filter which specifies one or more PerfMetricsType to include in the result; if none returns all. The resulting PerfSampleSeries are sorted by ids. May return any of the following canonical error codes: - NOT_FOUND - The containing Step does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPerfSampleSeriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.perfSampleSeries.list', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=['filter'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfSampleSeries', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesListRequest', response_type_name='ListPerfSampleSeriesResponse', supports_download=False)