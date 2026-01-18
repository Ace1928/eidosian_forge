from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsDagsTasksService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_dags_tasks resource."""
    _NAME = 'projects_locations_environments_dags_tasks'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsDagsTasksService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists tasks of a DAG.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsTasksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTasksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}/tasks', http_method='GET', method_id='composer.projects.locations.environments.dags.tasks.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/tasks', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsTasksListRequest', response_type_name='ListTasksResponse', supports_download=False)