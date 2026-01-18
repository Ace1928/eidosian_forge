from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps_perfSampleSeries_samples resource."""
    _NAME = 'projects_histories_executions_steps_perfSampleSeries_samples'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates a batch of PerfSamples - a client can submit multiple batches of Perf Samples through repeated calls to this method in order to split up a large request payload - duplicates and existing timestamp entries will be ignored. - the batch operation may partially succeed - the set of elements successfully inserted is returned in the response (omits items which already existed in the database). May return any of the following canonical error codes: - NOT_FOUND - The containing PerfSampleSeries does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchCreatePerfSamplesResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.steps.perfSampleSeries.samples.batchCreate', ordered_params=['projectId', 'historyId', 'executionId', 'stepId', 'sampleSeriesId'], path_params=['executionId', 'historyId', 'projectId', 'sampleSeriesId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfSampleSeries/{sampleSeriesId}/samples:batchCreate', request_field='batchCreatePerfSamplesRequest', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesBatchCreateRequest', response_type_name='BatchCreatePerfSamplesResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Performance Samples of a given Sample Series - The list results are sorted by timestamps ascending - The default page size is 500 samples; and maximum size allowed 5000 - The response token indicates the last returned PerfSample timestamp - When the results size exceeds the page size, submit a subsequent request including the page token to return the rest of the samples up to the page limit May return any of the following canonical error codes: - OUT_OF_RANGE - The specified request page_token is out of valid range - NOT_FOUND - The containing PerfSampleSeries does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPerfSamplesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.perfSampleSeries.samples.list', ordered_params=['projectId', 'historyId', 'executionId', 'stepId', 'sampleSeriesId'], path_params=['executionId', 'historyId', 'projectId', 'sampleSeriesId', 'stepId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfSampleSeries/{sampleSeriesId}/samples', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesListRequest', response_type_name='ListPerfSamplesResponse', supports_download=False)