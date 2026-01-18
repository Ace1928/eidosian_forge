from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsLocationPrometheusApiV1LabelService(base_api.BaseApiService):
    """Service class for the projects_location_prometheus_api_v1_label resource."""
    _NAME = 'projects_location_prometheus_api_v1_label'

    def __init__(self, client):
        super(MonitoringV1.ProjectsLocationPrometheusApiV1LabelService, self).__init__(client)
        self._upload_configs = {}

    def Values(self, request, global_params=None):
        """Lists possible values for a given label name.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1LabelValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('Values')
        return self._RunMethod(config, request, global_params=global_params)
    Values.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/label/{label}/values', http_method='GET', method_id='monitoring.projects.location.prometheus.api.v1.label.values', ordered_params=['name', 'location', 'label'], path_params=['label', 'location', 'name'], query_params=['end', 'match', 'start'], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/label/{label}/values', request_field='', request_type_name='MonitoringProjectsLocationPrometheusApiV1LabelValuesRequest', response_type_name='HttpBody', supports_download=False)