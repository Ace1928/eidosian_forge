from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAwsClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_awsClusters resource."""
    _NAME = 'projects_locations_awsClusters'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAwsClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AwsCluster resource on a given Google Cloud Platform project and region. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters', http_method='POST', method_id='gkemulticloud.projects.locations.awsClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['awsClusterId', 'validateOnly'], relative_path='v1/{+parent}/awsClusters', request_field='googleCloudGkemulticloudV1AwsCluster', request_type_name='GkemulticloudProjectsLocationsAwsClustersCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific AwsCluster resource. Fails if the cluster has one or more associated AwsNodePool resources. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}', http_method='DELETE', method_id='gkemulticloud.projects.locations.awsClusters.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GenerateAwsAccessToken(self, request, global_params=None):
        """Generates a short-lived access token to authenticate to a given AwsCluster resource.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGenerateAwsAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAwsAccessTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateAwsAccessToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateAwsAccessToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}:generateAwsAccessToken', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.generateAwsAccessToken', ordered_params=['awsCluster'], path_params=['awsCluster'], query_params=[], relative_path='v1/{+awsCluster}:generateAwsAccessToken', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersGenerateAwsAccessTokenRequest', response_type_name='GoogleCloudGkemulticloudV1GenerateAwsAccessTokenResponse', supports_download=False)

    def GenerateAwsClusterAgentToken(self, request, global_params=None):
        """Generates an access token for a cluster agent.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGenerateAwsClusterAgentTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAwsClusterAgentTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateAwsClusterAgentToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateAwsClusterAgentToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}:generateAwsClusterAgentToken', http_method='POST', method_id='gkemulticloud.projects.locations.awsClusters.generateAwsClusterAgentToken', ordered_params=['awsCluster'], path_params=['awsCluster'], query_params=[], relative_path='v1/{+awsCluster}:generateAwsClusterAgentToken', request_field='googleCloudGkemulticloudV1GenerateAwsClusterAgentTokenRequest', request_type_name='GkemulticloudProjectsLocationsAwsClustersGenerateAwsClusterAgentTokenRequest', response_type_name='GoogleCloudGkemulticloudV1GenerateAwsClusterAgentTokenResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Describes a specific AwsCluster resource.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AwsCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersGetRequest', response_type_name='GoogleCloudGkemulticloudV1AwsCluster', supports_download=False)

    def GetJwks(self, request, global_params=None):
        """Gets the public component of the cluster signing keys in JSON Web Key format.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGetJwksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AwsJsonWebKeys) The response message.
      """
        config = self.GetMethodConfig('GetJwks')
        return self._RunMethod(config, request, global_params=global_params)
    GetJwks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/jwks', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.getJwks', ordered_params=['awsCluster'], path_params=['awsCluster'], query_params=[], relative_path='v1/{+awsCluster}/jwks', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersGetJwksRequest', response_type_name='GoogleCloudGkemulticloudV1AwsJsonWebKeys', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AwsCluster resources on a given Google Cloud project and region.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1ListAwsClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/awsClusters', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersListRequest', response_type_name='GoogleCloudGkemulticloudV1ListAwsClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an AwsCluster.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}', http_method='PATCH', method_id='gkemulticloud.projects.locations.awsClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudGkemulticloudV1AwsCluster', request_type_name='GkemulticloudProjectsLocationsAwsClustersPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)