from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsDagsService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_dags resource."""
    _NAME = 'projects_locations_environments_dags'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsDagsService, self).__init__(client)
        self._upload_configs = {}

    def Activate(self, request, global_params=None):
        """Activates a dag.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsActivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dag) The response message.
      """
        config = self.GetMethodConfig('Activate')
        return self._RunMethod(config, request, global_params=global_params)
    Activate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}:activate', http_method='POST', method_id='composer.projects.locations.environments.dags.activate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:activate', request_field='activateDagRequest', request_type_name='ComposerProjectsLocationsEnvironmentsDagsActivateRequest', response_type_name='Dag', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a DAG.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dag) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}', http_method='GET', method_id='composer.projects.locations.environments.dags.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsGetRequest', response_type_name='Dag', supports_download=False)

    def GetSourceCode(self, request, global_params=None):
        """Retrieves DAG source code.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsGetSourceCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SourceCode) The response message.
      """
        config = self.GetMethodConfig('GetSourceCode')
        return self._RunMethod(config, request, global_params=global_params)
    GetSourceCode.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}:getSourceCode', http_method='GET', method_id='composer.projects.locations.environments.dags.getSourceCode', ordered_params=['dag'], path_params=['dag'], query_params=[], relative_path='v1alpha2/{+dag}:getSourceCode', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsGetSourceCodeRequest', response_type_name='SourceCode', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DAGs in an environment.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags', http_method='GET', method_id='composer.projects.locations.environments.dags.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/dags', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsListRequest', response_type_name='ListDagsResponse', supports_download=False)

    def ListStats(self, request, global_params=None):
        """List DAGs with statistics for a given time interval.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsListStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDagStatsResponse) The response message.
      """
        config = self.GetMethodConfig('ListStats')
        return self._RunMethod(config, request, global_params=global_params)
    ListStats.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags:listStats', http_method='GET', method_id='composer.projects.locations.environments.dags.listStats', ordered_params=['environment'], path_params=['environment'], query_params=['interval_endTime', 'interval_startTime', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+environment}/dags:listStats', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsListStatsRequest', response_type_name='ListDagStatsResponse', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses a dag.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dag) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}:pause', http_method='POST', method_id='composer.projects.locations.environments.dags.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:pause', request_field='pauseDagRequest', request_type_name='ComposerProjectsLocationsEnvironmentsDagsPauseRequest', response_type_name='Dag', supports_download=False)

    def Trigger(self, request, global_params=None):
        """Trigger a DAG run.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsTriggerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DagRun) The response message.
      """
        config = self.GetMethodConfig('Trigger')
        return self._RunMethod(config, request, global_params=global_params)
    Trigger.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}:trigger', http_method='POST', method_id='composer.projects.locations.environments.dags.trigger', ordered_params=['dag'], path_params=['dag'], query_params=[], relative_path='v1alpha2/{+dag}:trigger', request_field='triggerDagRequest', request_type_name='ComposerProjectsLocationsEnvironmentsDagsTriggerRequest', response_type_name='DagRun', supports_download=False)