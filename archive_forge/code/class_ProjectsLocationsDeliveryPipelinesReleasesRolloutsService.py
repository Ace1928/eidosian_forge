from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsDeliveryPipelinesReleasesRolloutsService(base_api.BaseApiService):
    """Service class for the projects_locations_deliveryPipelines_releases_rollouts resource."""
    _NAME = 'projects_locations_deliveryPipelines_releases_rollouts'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsDeliveryPipelinesReleasesRolloutsService, self).__init__(client)
        self._upload_configs = {}

    def Advance(self, request, global_params=None):
        """Advances a Rollout in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsAdvanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AdvanceRolloutResponse) The response message.
      """
        config = self.GetMethodConfig('Advance')
        return self._RunMethod(config, request, global_params=global_params)
    Advance.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}:advance', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.advance', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:advance', request_field='advanceRolloutRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsAdvanceRequest', response_type_name='AdvanceRolloutResponse', supports_download=False)

    def Approve(self, request, global_params=None):
        """Approves a Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApproveRolloutResponse) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}:approve', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveRolloutRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsApproveRequest', response_type_name='ApproveRolloutResponse', supports_download=False)

    def Cancel(self, request, global_params=None):
        """Cancels a Rollout in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CancelRolloutResponse) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}:cancel', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelRolloutRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCancelRequest', response_type_name='CancelRolloutResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Rollout in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.create', ordered_params=['parent'], path_params=['parent'], query_params=['overrideDeployPolicy', 'requestId', 'rolloutId', 'startingPhaseId', 'validateOnly'], relative_path='v1/{+parent}/rollouts', request_field='rollout', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Rollout) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsGetRequest', response_type_name='Rollout', supports_download=False)

    def IgnoreJob(self, request, global_params=None):
        """Ignores the specified Job in a Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsIgnoreJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IgnoreJobResponse) The response message.
      """
        config = self.GetMethodConfig('IgnoreJob')
        return self._RunMethod(config, request, global_params=global_params)
    IgnoreJob.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}:ignoreJob', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.ignoreJob', ordered_params=['rollout'], path_params=['rollout'], query_params=[], relative_path='v1/{+rollout}:ignoreJob', request_field='ignoreJobRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsIgnoreJobRequest', response_type_name='IgnoreJobResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Rollouts in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRolloutsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/rollouts', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsListRequest', response_type_name='ListRolloutsResponse', supports_download=False)

    def RetryJob(self, request, global_params=None):
        """Retries the specified Job in a Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsRetryJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetryJobResponse) The response message.
      """
        config = self.GetMethodConfig('RetryJob')
        return self._RunMethod(config, request, global_params=global_params)
    RetryJob.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}/rollouts/{rolloutsId}:retryJob', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts.retryJob', ordered_params=['rollout'], path_params=['rollout'], query_params=[], relative_path='v1/{+rollout}:retryJob', request_field='retryJobRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsRetryJobRequest', response_type_name='RetryJobResponse', supports_download=False)