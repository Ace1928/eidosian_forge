from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsIndexEndpointsService(base_api.BaseApiService):
    """Service class for the projects_locations_indexEndpoints resource."""
    _NAME = 'projects_locations_indexEndpoints'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsIndexEndpointsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an IndexEndpoint.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints', http_method='POST', method_id='aiplatform.projects.locations.indexEndpoints.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/indexEndpoints', request_field='googleCloudAiplatformV1IndexEndpoint', request_type_name='AiplatformProjectsLocationsIndexEndpointsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an IndexEndpoint.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}', http_method='DELETE', method_id='aiplatform.projects.locations.indexEndpoints.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsIndexEndpointsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def DeployIndex(self, request, global_params=None):
        """Deploys an Index into this IndexEndpoint, creating a DeployedIndex within it. Only non-empty Indexes can be deployed.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsDeployIndexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('DeployIndex')
        return self._RunMethod(config, request, global_params=global_params)
    DeployIndex.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}:deployIndex', http_method='POST', method_id='aiplatform.projects.locations.indexEndpoints.deployIndex', ordered_params=['indexEndpoint'], path_params=['indexEndpoint'], query_params=[], relative_path='v1/{+indexEndpoint}:deployIndex', request_field='googleCloudAiplatformV1DeployIndexRequest', request_type_name='AiplatformProjectsLocationsIndexEndpointsDeployIndexRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an IndexEndpoint.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1IndexEndpoint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}', http_method='GET', method_id='aiplatform.projects.locations.indexEndpoints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsIndexEndpointsGetRequest', response_type_name='GoogleCloudAiplatformV1IndexEndpoint', supports_download=False)

    def List(self, request, global_params=None):
        """Lists IndexEndpoints in a Location.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListIndexEndpointsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints', http_method='GET', method_id='aiplatform.projects.locations.indexEndpoints.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/indexEndpoints', request_field='', request_type_name='AiplatformProjectsLocationsIndexEndpointsListRequest', response_type_name='GoogleCloudAiplatformV1ListIndexEndpointsResponse', supports_download=False)

    def MutateDeployedIndex(self, request, global_params=None):
        """Update an existing DeployedIndex under an IndexEndpoint.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsMutateDeployedIndexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('MutateDeployedIndex')
        return self._RunMethod(config, request, global_params=global_params)
    MutateDeployedIndex.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}:mutateDeployedIndex', http_method='POST', method_id='aiplatform.projects.locations.indexEndpoints.mutateDeployedIndex', ordered_params=['indexEndpoint'], path_params=['indexEndpoint'], query_params=[], relative_path='v1/{+indexEndpoint}:mutateDeployedIndex', request_field='googleCloudAiplatformV1DeployedIndex', request_type_name='AiplatformProjectsLocationsIndexEndpointsMutateDeployedIndexRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an IndexEndpoint.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1IndexEndpoint) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}', http_method='PATCH', method_id='aiplatform.projects.locations.indexEndpoints.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1IndexEndpoint', request_type_name='AiplatformProjectsLocationsIndexEndpointsPatchRequest', response_type_name='GoogleCloudAiplatformV1IndexEndpoint', supports_download=False)

    def UndeployIndex(self, request, global_params=None):
        """Undeploys an Index from an IndexEndpoint, removing a DeployedIndex from it, and freeing all resources it's using.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('UndeployIndex')
        return self._RunMethod(config, request, global_params=global_params)
    UndeployIndex.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexEndpoints/{indexEndpointsId}:undeployIndex', http_method='POST', method_id='aiplatform.projects.locations.indexEndpoints.undeployIndex', ordered_params=['indexEndpoint'], path_params=['indexEndpoint'], query_params=[], relative_path='v1/{+indexEndpoint}:undeployIndex', request_field='googleCloudAiplatformV1UndeployIndexRequest', request_type_name='AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)