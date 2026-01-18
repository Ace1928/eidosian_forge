from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class ProjectsMetricsService(base_api.BaseApiService):
    """Service class for the projects_metrics resource."""
    _NAME = 'projects_metrics'

    def __init__(self, client):
        super(LoggingV2.ProjectsMetricsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a logs-based metric.

      Args:
        request: (LoggingProjectsMetricsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogMetric) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/metrics', http_method='POST', method_id='logging.projects.metrics.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/metrics', request_field='logMetric', request_type_name='LoggingProjectsMetricsCreateRequest', response_type_name='LogMetric', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a logs-based metric.

      Args:
        request: (LoggingProjectsMetricsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/metrics/{metricsId}', http_method='DELETE', method_id='logging.projects.metrics.delete', ordered_params=['metricName'], path_params=['metricName'], query_params=[], relative_path='v2/{+metricName}', request_field='', request_type_name='LoggingProjectsMetricsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a logs-based metric.

      Args:
        request: (LoggingProjectsMetricsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogMetric) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/metrics/{metricsId}', http_method='GET', method_id='logging.projects.metrics.get', ordered_params=['metricName'], path_params=['metricName'], query_params=[], relative_path='v2/{+metricName}', request_field='', request_type_name='LoggingProjectsMetricsGetRequest', response_type_name='LogMetric', supports_download=False)

    def List(self, request, global_params=None):
        """Lists logs-based metrics.

      Args:
        request: (LoggingProjectsMetricsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLogMetricsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/metrics', http_method='GET', method_id='logging.projects.metrics.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/metrics', request_field='', request_type_name='LoggingProjectsMetricsListRequest', response_type_name='ListLogMetricsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Creates or updates a logs-based metric.

      Args:
        request: (LoggingProjectsMetricsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogMetric) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/metrics/{metricsId}', http_method='PUT', method_id='logging.projects.metrics.update', ordered_params=['metricName'], path_params=['metricName'], query_params=[], relative_path='v2/{+metricName}', request_field='logMetric', request_type_name='LoggingProjectsMetricsUpdateRequest', response_type_name='LogMetric', supports_download=False)