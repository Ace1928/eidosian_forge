from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
class ProjectsLocationsRolloutsService(base_api.BaseApiService):
    """Service class for the projects_locations_rollouts resource."""
    _NAME = 'projects_locations_rollouts'

    def __init__(self, client):
        super(GkehubV1alpha.ProjectsLocationsRolloutsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new rollout resource.

      Args:
        request: (GkehubProjectsLocationsRolloutsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts', http_method='POST', method_id='gkehub.projects.locations.rollouts.create', ordered_params=['parent'], path_params=['parent'], query_params=['rolloutId'], relative_path='v1alpha/{+parent}/rollouts', request_field='rollout', request_type_name='GkehubProjectsLocationsRolloutsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a Rollout.

      Args:
        request: (GkehubProjectsLocationsRolloutsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts/{rolloutsId}', http_method='DELETE', method_id='gkehub.projects.locations.rollouts.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='GkehubProjectsLocationsRolloutsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a single rollout.

      Args:
        request: (GkehubProjectsLocationsRolloutsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Rollout) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts/{rolloutsId}', http_method='GET', method_id='gkehub.projects.locations.rollouts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='GkehubProjectsLocationsRolloutsGetRequest', response_type_name='Rollout', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve the list of all rollouts.

      Args:
        request: (GkehubProjectsLocationsRolloutsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRolloutsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts', http_method='GET', method_id='gkehub.projects.locations.rollouts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/rollouts', request_field='', request_type_name='GkehubProjectsLocationsRolloutsListRequest', response_type_name='ListRolloutsResponse', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pause a running Rollout. The rollout will not be started on new clusters, however the rollout running on the cluster will be allowed to finish.

      Args:
        request: (GkehubProjectsLocationsRolloutsPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts/{rolloutsId}:pause', http_method='POST', method_id='gkehub.projects.locations.rollouts.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:pause', request_field='pauseRolloutRequest', request_type_name='GkehubProjectsLocationsRolloutsPauseRequest', response_type_name='Operation', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resume a paused Rollout. The rollout will be resumed and allowed to be started on clusters.

      Args:
        request: (GkehubProjectsLocationsRolloutsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/rollouts/{rolloutsId}:resume', http_method='POST', method_id='gkehub.projects.locations.rollouts.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:resume', request_field='resumeRolloutRequest', request_type_name='GkehubProjectsLocationsRolloutsResumeRequest', response_type_name='Operation', supports_download=False)