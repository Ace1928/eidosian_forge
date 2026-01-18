from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsDagsDagRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_dags_dagRuns resource."""
    _NAME = 'projects_locations_environments_dags_dagRuns'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsDagsDagRunsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a DAG run.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsDagRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DagRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}/dagRuns/{dagRunsId}', http_method='GET', method_id='composer.projects.locations.environments.dags.dagRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsDagRunsGetRequest', response_type_name='DagRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DAG runs of a DAG.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsDagRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDagRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}/dagRuns', http_method='GET', method_id='composer.projects.locations.environments.dags.dagRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/dagRuns', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsDagRunsListRequest', response_type_name='ListDagRunsResponse', supports_download=False)