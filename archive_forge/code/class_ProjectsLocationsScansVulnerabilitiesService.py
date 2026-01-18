from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ondemandscanning.v1 import ondemandscanning_v1_messages as messages
class ProjectsLocationsScansVulnerabilitiesService(base_api.BaseApiService):
    """Service class for the projects_locations_scans_vulnerabilities resource."""
    _NAME = 'projects_locations_scans_vulnerabilities'

    def __init__(self, client):
        super(OndemandscanningV1.ProjectsLocationsScansVulnerabilitiesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists vulnerabilities resulting from a successfully completed scan.

      Args:
        request: (OndemandscanningProjectsLocationsScansVulnerabilitiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVulnerabilitiesResponseV1) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/scans/{scansId}/vulnerabilities', http_method='GET', method_id='ondemandscanning.projects.locations.scans.vulnerabilities.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/vulnerabilities', request_field='', request_type_name='OndemandscanningProjectsLocationsScansVulnerabilitiesListRequest', response_type_name='ListVulnerabilitiesResponseV1', supports_download=False)