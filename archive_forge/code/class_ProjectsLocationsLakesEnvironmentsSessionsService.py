from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesEnvironmentsSessionsService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_environments_sessions resource."""
    _NAME = 'projects_locations_lakes_environments_sessions'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesEnvironmentsSessionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists session resources in an environment.

      Args:
        request: (DataplexProjectsLocationsLakesEnvironmentsSessionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/environments/{environmentsId}/sessions', http_method='GET', method_id='dataplex.projects.locations.lakes.environments.sessions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sessions', request_field='', request_type_name='DataplexProjectsLocationsLakesEnvironmentsSessionsListRequest', response_type_name='GoogleCloudDataplexV1ListSessionsResponse', supports_download=False)