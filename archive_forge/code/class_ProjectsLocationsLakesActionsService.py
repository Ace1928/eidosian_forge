from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesActionsService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_actions resource."""
    _NAME = 'projects_locations_lakes_actions'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesActionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists action resources in a lake.

      Args:
        request: (DataplexProjectsLocationsLakesActionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListActionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/actions', http_method='GET', method_id='dataplex.projects.locations.lakes.actions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/actions', request_field='', request_type_name='DataplexProjectsLocationsLakesActionsListRequest', response_type_name='GoogleCloudDataplexV1ListActionsResponse', supports_download=False)