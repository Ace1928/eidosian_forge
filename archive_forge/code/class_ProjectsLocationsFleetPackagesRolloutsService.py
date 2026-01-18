from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.configdelivery.v1alpha import configdelivery_v1alpha_messages as messages
class ProjectsLocationsFleetPackagesRolloutsService(base_api.BaseApiService):
    """Service class for the projects_locations_fleetPackages_rollouts resource."""
    _NAME = 'projects_locations_fleetPackages_rollouts'

    def __init__(self, client):
        super(ConfigdeliveryV1alpha.ProjectsLocationsFleetPackagesRolloutsService, self).__init__(client)
        self._upload_configs = {}

    def Abort(self, request, global_params=None):
        """Abort a Rollout.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsAbortRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Abort')
        return self._RunMethod(config, request, global_params=global_params)
    Abort.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}/rollouts/{rolloutsId}:abort', http_method='POST', method_id='configdelivery.projects.locations.fleetPackages.rollouts.abort', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:abort', request_field='abortRolloutRequest', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesRolloutsAbortRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Rollout.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Rollout) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}/rollouts/{rolloutsId}', http_method='GET', method_id='configdelivery.projects.locations.fleetPackages.rollouts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesRolloutsGetRequest', response_type_name='Rollout', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Rollouts in a given project, location, and fleet package.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRolloutsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}/rollouts', http_method='GET', method_id='configdelivery.projects.locations.fleetPackages.rollouts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/rollouts', request_field='', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesRolloutsListRequest', response_type_name='ListRolloutsResponse', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resume a Rollout.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}/rollouts/{rolloutsId}:resume', http_method='POST', method_id='configdelivery.projects.locations.fleetPackages.rollouts.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:resume', request_field='resumeRolloutRequest', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesRolloutsResumeRequest', response_type_name='Operation', supports_download=False)

    def Suspend(self, request, global_params=None):
        """Suspend a Rollout.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsSuspendRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Suspend')
        return self._RunMethod(config, request, global_params=global_params)
    Suspend.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}/rollouts/{rolloutsId}:suspend', http_method='POST', method_id='configdelivery.projects.locations.fleetPackages.rollouts.suspend', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:suspend', request_field='suspendRolloutRequest', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesRolloutsSuspendRequest', response_type_name='Operation', supports_download=False)