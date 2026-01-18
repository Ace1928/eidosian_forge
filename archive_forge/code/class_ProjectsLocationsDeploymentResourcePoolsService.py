from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDeploymentResourcePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_deploymentResourcePools resource."""
    _NAME = 'projects_locations_deploymentResourcePools'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDeploymentResourcePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a DeploymentResourcePool.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deploymentResourcePools', http_method='POST', method_id='aiplatform.projects.locations.deploymentResourcePools.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/deploymentResourcePools', request_field='googleCloudAiplatformV1CreateDeploymentResourcePoolRequest', request_type_name='AiplatformProjectsLocationsDeploymentResourcePoolsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a DeploymentResourcePool.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deploymentResourcePools/{deploymentResourcePoolsId}', http_method='DELETE', method_id='aiplatform.projects.locations.deploymentResourcePools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDeploymentResourcePoolsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a DeploymentResourcePool.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1DeploymentResourcePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deploymentResourcePools/{deploymentResourcePoolsId}', http_method='GET', method_id='aiplatform.projects.locations.deploymentResourcePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDeploymentResourcePoolsGetRequest', response_type_name='GoogleCloudAiplatformV1DeploymentResourcePool', supports_download=False)

    def List(self, request, global_params=None):
        """List DeploymentResourcePools in a location.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListDeploymentResourcePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deploymentResourcePools', http_method='GET', method_id='aiplatform.projects.locations.deploymentResourcePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/deploymentResourcePools', request_field='', request_type_name='AiplatformProjectsLocationsDeploymentResourcePoolsListRequest', response_type_name='GoogleCloudAiplatformV1ListDeploymentResourcePoolsResponse', supports_download=False)

    def QueryDeployedModels(self, request, global_params=None):
        """List DeployedModels that have been deployed on this DeploymentResourcePool.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsQueryDeployedModelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1QueryDeployedModelsResponse) The response message.
      """
        config = self.GetMethodConfig('QueryDeployedModels')
        return self._RunMethod(config, request, global_params=global_params)
    QueryDeployedModels.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deploymentResourcePools/{deploymentResourcePoolsId}:queryDeployedModels', http_method='GET', method_id='aiplatform.projects.locations.deploymentResourcePools.queryDeployedModels', ordered_params=['deploymentResourcePool'], path_params=['deploymentResourcePool'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+deploymentResourcePool}:queryDeployedModels', request_field='', request_type_name='AiplatformProjectsLocationsDeploymentResourcePoolsQueryDeployedModelsRequest', response_type_name='GoogleCloudAiplatformV1QueryDeployedModelsResponse', supports_download=False)