from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsSessionsSparkApplicationsService(base_api.BaseApiService):
    """Service class for the projects_locations_sessions_sparkApplications resource."""
    _NAME = 'projects_locations_sessions_sparkApplications'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsSessionsSparkApplicationsService, self).__init__(client)
        self._upload_configs = {}

    def Access(self, request, global_params=None):
        """Obtain high level information corresponding to a single Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationResponse) The response message.
      """
        config = self.GetMethodConfig('Access')
        return self._RunMethod(config, request, global_params=global_params)
    Access.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:access', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.access', ordered_params=['name'], path_params=['name'], query_params=['parent'], relative_path='v1/{+name}:access', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessRequest', response_type_name='AccessSessionSparkApplicationResponse', supports_download=False)

    def AccessEnvironmentInfo(self, request, global_params=None):
        """Obtain environment details for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessEnvironmentInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationEnvironmentInfoResponse) The response message.
      """
        config = self.GetMethodConfig('AccessEnvironmentInfo')
        return self._RunMethod(config, request, global_params=global_params)
    AccessEnvironmentInfo.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessEnvironmentInfo', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessEnvironmentInfo', ordered_params=['name'], path_params=['name'], query_params=['parent'], relative_path='v1/{+name}:accessEnvironmentInfo', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessEnvironmentInfoRequest', response_type_name='AccessSessionSparkApplicationEnvironmentInfoResponse', supports_download=False)

    def AccessJob(self, request, global_params=None):
        """Obtain data corresponding to a spark job for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationJobResponse) The response message.
      """
        config = self.GetMethodConfig('AccessJob')
        return self._RunMethod(config, request, global_params=global_params)
    AccessJob.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessJob', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessJob', ordered_params=['name'], path_params=['name'], query_params=['jobId', 'parent'], relative_path='v1/{+name}:accessJob', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessJobRequest', response_type_name='AccessSessionSparkApplicationJobResponse', supports_download=False)

    def AccessSqlPlan(self, request, global_params=None):
        """Obtain Spark Plan Graph for a Spark Application SQL execution. Limits the number of clusters returned as part of the graph to 10000.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlPlanRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationSqlSparkPlanGraphResponse) The response message.
      """
        config = self.GetMethodConfig('AccessSqlPlan')
        return self._RunMethod(config, request, global_params=global_params)
    AccessSqlPlan.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessSqlPlan', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessSqlPlan', ordered_params=['name'], path_params=['name'], query_params=['executionId', 'parent'], relative_path='v1/{+name}:accessSqlPlan', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlPlanRequest', response_type_name='AccessSessionSparkApplicationSqlSparkPlanGraphResponse', supports_download=False)

    def AccessSqlQuery(self, request, global_params=None):
        """Obtain data corresponding to a particular SQL Query for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationSqlQueryResponse) The response message.
      """
        config = self.GetMethodConfig('AccessSqlQuery')
        return self._RunMethod(config, request, global_params=global_params)
    AccessSqlQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessSqlQuery', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessSqlQuery', ordered_params=['name'], path_params=['name'], query_params=['details', 'executionId', 'parent', 'planDescription'], relative_path='v1/{+name}:accessSqlQuery', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlQueryRequest', response_type_name='AccessSessionSparkApplicationSqlQueryResponse', supports_download=False)

    def AccessStageAttempt(self, request, global_params=None):
        """Obtain data corresponding to a spark stage attempt for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessStageAttemptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationStageAttemptResponse) The response message.
      """
        config = self.GetMethodConfig('AccessStageAttempt')
        return self._RunMethod(config, request, global_params=global_params)
    AccessStageAttempt.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessStageAttempt', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessStageAttempt', ordered_params=['name'], path_params=['name'], query_params=['parent', 'stageAttemptId', 'stageId', 'summaryMetricsMask'], relative_path='v1/{+name}:accessStageAttempt', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessStageAttemptRequest', response_type_name='AccessSessionSparkApplicationStageAttemptResponse', supports_download=False)

    def AccessStageRddGraph(self, request, global_params=None):
        """Obtain RDD operation graph for a Spark Application Stage. Limits the number of clusters returned as part of the graph to 10000.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessStageRddGraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationStageRddOperationGraphResponse) The response message.
      """
        config = self.GetMethodConfig('AccessStageRddGraph')
        return self._RunMethod(config, request, global_params=global_params)
    AccessStageRddGraph.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:accessStageRddGraph', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.accessStageRddGraph', ordered_params=['name'], path_params=['name'], query_params=['parent', 'stageId'], relative_path='v1/{+name}:accessStageRddGraph', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsAccessStageRddGraphRequest', response_type_name='AccessSessionSparkApplicationStageRddOperationGraphResponse', supports_download=False)

    def Search(self, request, global_params=None):
        """Obtain high level information and list of Spark Applications corresponding to a batch.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications:search', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.search', ordered_params=['parent'], path_params=['parent'], query_params=['applicationStatus', 'maxEndTime', 'maxTime', 'minEndTime', 'minTime', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sparkApplications:search', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchRequest', response_type_name='SearchSessionSparkApplicationsResponse', supports_download=False)

    def SearchExecutorStageSummary(self, request, global_params=None):
        """Obtain executor summary with respect to a spark stage attempt.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorStageSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationExecutorStageSummaryResponse) The response message.
      """
        config = self.GetMethodConfig('SearchExecutorStageSummary')
        return self._RunMethod(config, request, global_params=global_params)
    SearchExecutorStageSummary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchExecutorStageSummary', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchExecutorStageSummary', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken', 'parent', 'stageAttemptId', 'stageId'], relative_path='v1/{+name}:searchExecutorStageSummary', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorStageSummaryRequest', response_type_name='SearchSessionSparkApplicationExecutorStageSummaryResponse', supports_download=False)

    def SearchExecutors(self, request, global_params=None):
        """Obtain data corresponding to executors for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationExecutorsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchExecutors')
        return self._RunMethod(config, request, global_params=global_params)
    SearchExecutors.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchExecutors', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchExecutors', ordered_params=['name'], path_params=['name'], query_params=['executorStatus', 'pageSize', 'pageToken', 'parent'], relative_path='v1/{+name}:searchExecutors', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorsRequest', response_type_name='SearchSessionSparkApplicationExecutorsResponse', supports_download=False)

    def SearchJobs(self, request, global_params=None):
        """Obtain list of spark jobs corresponding to a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationJobsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchJobs')
        return self._RunMethod(config, request, global_params=global_params)
    SearchJobs.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchJobs', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchJobs', ordered_params=['name'], path_params=['name'], query_params=['jobStatus', 'pageSize', 'pageToken', 'parent'], relative_path='v1/{+name}:searchJobs', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchJobsRequest', response_type_name='SearchSessionSparkApplicationJobsResponse', supports_download=False)

    def SearchSqlQueries(self, request, global_params=None):
        """Obtain data corresponding to SQL Queries for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchSqlQueriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationSqlQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchSqlQueries')
        return self._RunMethod(config, request, global_params=global_params)
    SearchSqlQueries.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchSqlQueries', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchSqlQueries', ordered_params=['name'], path_params=['name'], query_params=['details', 'pageSize', 'pageToken', 'parent', 'planDescription'], relative_path='v1/{+name}:searchSqlQueries', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchSqlQueriesRequest', response_type_name='SearchSessionSparkApplicationSqlQueriesResponse', supports_download=False)

    def SearchStageAttemptTasks(self, request, global_params=None):
        """Obtain data corresponding to tasks for a spark stage attempt for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptTasksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStageAttemptTasksResponse) The response message.
      """
        config = self.GetMethodConfig('SearchStageAttemptTasks')
        return self._RunMethod(config, request, global_params=global_params)
    SearchStageAttemptTasks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchStageAttemptTasks', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchStageAttemptTasks', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken', 'parent', 'sortRuntime', 'stageAttemptId', 'stageId', 'taskStatus'], relative_path='v1/{+name}:searchStageAttemptTasks', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptTasksRequest', response_type_name='SearchSessionSparkApplicationStageAttemptTasksResponse', supports_download=False)

    def SearchStageAttempts(self, request, global_params=None):
        """Obtain data corresponding to a spark stage attempts for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStageAttemptsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchStageAttempts')
        return self._RunMethod(config, request, global_params=global_params)
    SearchStageAttempts.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchStageAttempts', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchStageAttempts', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken', 'parent', 'stageId', 'summaryMetricsMask'], relative_path='v1/{+name}:searchStageAttempts', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptsRequest', response_type_name='SearchSessionSparkApplicationStageAttemptsResponse', supports_download=False)

    def SearchStages(self, request, global_params=None):
        """Obtain data corresponding to stages for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStagesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchStages')
        return self._RunMethod(config, request, global_params=global_params)
    SearchStages.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:searchStages', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.searchStages', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken', 'parent', 'stageStatus', 'summaryMetricsMask'], relative_path='v1/{+name}:searchStages', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSearchStagesRequest', response_type_name='SearchSessionSparkApplicationStagesResponse', supports_download=False)

    def SummarizeExecutors(self, request, global_params=None):
        """Obtain summary of Executor Summary for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeExecutorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationExecutorsResponse) The response message.
      """
        config = self.GetMethodConfig('SummarizeExecutors')
        return self._RunMethod(config, request, global_params=global_params)
    SummarizeExecutors.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:summarizeExecutors', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.summarizeExecutors', ordered_params=['name'], path_params=['name'], query_params=['parent'], relative_path='v1/{+name}:summarizeExecutors', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSummarizeExecutorsRequest', response_type_name='SummarizeSessionSparkApplicationExecutorsResponse', supports_download=False)

    def SummarizeJobs(self, request, global_params=None):
        """Obtain summary of Jobs for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationJobsResponse) The response message.
      """
        config = self.GetMethodConfig('SummarizeJobs')
        return self._RunMethod(config, request, global_params=global_params)
    SummarizeJobs.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:summarizeJobs', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.summarizeJobs', ordered_params=['name'], path_params=['name'], query_params=['parent'], relative_path='v1/{+name}:summarizeJobs', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSummarizeJobsRequest', response_type_name='SummarizeSessionSparkApplicationJobsResponse', supports_download=False)

    def SummarizeStageAttemptTasks(self, request, global_params=None):
        """Obtain summary of Tasks for a Spark Application Stage Attempt.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStageAttemptTasksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationStageAttemptTasksResponse) The response message.
      """
        config = self.GetMethodConfig('SummarizeStageAttemptTasks')
        return self._RunMethod(config, request, global_params=global_params)
    SummarizeStageAttemptTasks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:summarizeStageAttemptTasks', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.summarizeStageAttemptTasks', ordered_params=['name'], path_params=['name'], query_params=['parent', 'stageAttemptId', 'stageId'], relative_path='v1/{+name}:summarizeStageAttemptTasks', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStageAttemptTasksRequest', response_type_name='SummarizeSessionSparkApplicationStageAttemptTasksResponse', supports_download=False)

    def SummarizeStages(self, request, global_params=None):
        """Obtain summary of Stages for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationStagesResponse) The response message.
      """
        config = self.GetMethodConfig('SummarizeStages')
        return self._RunMethod(config, request, global_params=global_params)
    SummarizeStages.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:summarizeStages', http_method='GET', method_id='dataproc.projects.locations.sessions.sparkApplications.summarizeStages', ordered_params=['name'], path_params=['name'], query_params=['parent'], relative_path='v1/{+name}:summarizeStages', request_field='', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStagesRequest', response_type_name='SummarizeSessionSparkApplicationStagesResponse', supports_download=False)

    def Write(self, request, global_params=None):
        """Write wrapper objects from dataplane to spanner.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsWriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WriteSessionSparkApplicationContextResponse) The response message.
      """
        config = self.GetMethodConfig('Write')
        return self._RunMethod(config, request, global_params=global_params)
    Write.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}/sparkApplications/{sparkApplicationsId}:write', http_method='POST', method_id='dataproc.projects.locations.sessions.sparkApplications.write', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:write', request_field='writeSessionSparkApplicationContextRequest', request_type_name='DataprocProjectsLocationsSessionsSparkApplicationsWriteRequest', response_type_name='WriteSessionSparkApplicationContextResponse', supports_download=False)