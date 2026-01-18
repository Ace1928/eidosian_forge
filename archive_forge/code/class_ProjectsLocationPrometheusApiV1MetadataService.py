from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsLocationPrometheusApiV1MetadataService(base_api.BaseApiService):
    """Service class for the projects_location_prometheus_api_v1_metadata resource."""
    _NAME = 'projects_location_prometheus_api_v1_metadata'

    def __init__(self, client):
        super(MonitoringV1.ProjectsLocationPrometheusApiV1MetadataService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists metadata for metrics.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1MetadataListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/metadata', http_method='GET', method_id='monitoring.projects.location.prometheus.api.v1.metadata.list', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=['limit', 'metric'], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/metadata', request_field='', request_type_name='MonitoringProjectsLocationPrometheusApiV1MetadataListRequest', response_type_name='HttpBody', supports_download=False)