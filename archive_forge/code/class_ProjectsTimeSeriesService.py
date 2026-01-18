from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsTimeSeriesService(base_api.BaseApiService):
    """Service class for the projects_timeSeries resource."""
    _NAME = 'projects_timeSeries'

    def __init__(self, client):
        super(MonitoringV3.ProjectsTimeSeriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates or adds data to one or more time series. The response is empty if all time series in the request were written. If any time series could not be written, a corresponding failure message is included in the error response. This method does not support resource locations constraint of an organization policy (https://cloud.google.com/resource-manager/docs/organization-policy/defining-locations#setting_the_organization_policy).

      Args:
        request: (MonitoringProjectsTimeSeriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/timeSeries', http_method='POST', method_id='monitoring.projects.timeSeries.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/timeSeries', request_field='createTimeSeriesRequest', request_type_name='MonitoringProjectsTimeSeriesCreateRequest', response_type_name='Empty', supports_download=False)

    def CreateService(self, request, global_params=None):
        """Creates or adds data to one or more service time series. A service time series is a time series for a metric from a Google Cloud service. The response is empty if all time series in the request were written. If any time series could not be written, a corresponding failure message is included in the error response. This endpoint rejects writes to user-defined metrics. This method is only for use by Google Cloud services. Use projects.timeSeries.create instead.

      Args:
        request: (MonitoringProjectsTimeSeriesCreateServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('CreateService')
        return self._RunMethod(config, request, global_params=global_params)
    CreateService.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/timeSeries:createService', http_method='POST', method_id='monitoring.projects.timeSeries.createService', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/timeSeries:createService', request_field='createTimeSeriesRequest', request_type_name='MonitoringProjectsTimeSeriesCreateServiceRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists time series that match a filter.

      Args:
        request: (MonitoringProjectsTimeSeriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTimeSeriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/timeSeries', http_method='GET', method_id='monitoring.projects.timeSeries.list', ordered_params=['name'], path_params=['name'], query_params=['aggregation_alignmentPeriod', 'aggregation_crossSeriesReducer', 'aggregation_groupByFields', 'aggregation_perSeriesAligner', 'filter', 'interval_endTime', 'interval_startTime', 'orderBy', 'pageSize', 'pageToken', 'secondaryAggregation_alignmentPeriod', 'secondaryAggregation_crossSeriesReducer', 'secondaryAggregation_groupByFields', 'secondaryAggregation_perSeriesAligner', 'view'], relative_path='v3/{+name}/timeSeries', request_field='', request_type_name='MonitoringProjectsTimeSeriesListRequest', response_type_name='ListTimeSeriesResponse', supports_download=False)

    def Query(self, request, global_params=None):
        """Queries time series using Monitoring Query Language.

      Args:
        request: (MonitoringProjectsTimeSeriesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryTimeSeriesResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/timeSeries:query', http_method='POST', method_id='monitoring.projects.timeSeries.query', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/timeSeries:query', request_field='queryTimeSeriesRequest', request_type_name='MonitoringProjectsTimeSeriesQueryRequest', response_type_name='QueryTimeSeriesResponse', supports_download=False)