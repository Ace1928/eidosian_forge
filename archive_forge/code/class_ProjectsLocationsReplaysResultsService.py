from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class ProjectsLocationsReplaysResultsService(base_api.BaseApiService):
    """Service class for the projects_locations_replays_results resource."""
    _NAME = 'projects_locations_replays_results'

    def __init__(self, client):
        super(PolicysimulatorV1beta.ProjectsLocationsReplaysResultsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the results of running a Replay.

      Args:
        request: (PolicysimulatorProjectsLocationsReplaysResultsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaListReplayResultsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/replays/{replaysId}/results', http_method='GET', method_id='policysimulator.projects.locations.replays.results.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/results', request_field='', request_type_name='PolicysimulatorProjectsLocationsReplaysResultsListRequest', response_type_name='GoogleCloudPolicysimulatorV1betaListReplayResultsResponse', supports_download=False)