from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsDeliveryPipelinesReleasesService(base_api.BaseApiService):
    """Service class for the projects_locations_deliveryPipelines_releases resource."""
    _NAME = 'projects_locations_deliveryPipelines_releases'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsDeliveryPipelinesReleasesService, self).__init__(client)
        self._upload_configs = {}

    def Abandon(self, request, global_params=None):
        """Abandons a Release in the Delivery Pipeline.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesAbandonRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AbandonReleaseResponse) The response message.
      """
        config = self.GetMethodConfig('Abandon')
        return self._RunMethod(config, request, global_params=global_params)
    Abandon.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}:abandon', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.abandon', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:abandon', request_field='abandonReleaseRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesAbandonRequest', response_type_name='AbandonReleaseResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Release in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.create', ordered_params=['parent'], path_params=['parent'], query_params=['overrideDeployPolicy', 'releaseId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/releases', request_field='release', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Release.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Release) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases/{releasesId}', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesGetRequest', response_type_name='Release', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Releases in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReleasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/releases', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.releases.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/releases', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesReleasesListRequest', response_type_name='ListReleasesResponse', supports_download=False)