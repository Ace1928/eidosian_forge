from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAttachedClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_attachedClusters resource."""
    _NAME = 'projects_locations_attachedClusters'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAttachedClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AttachedCluster resource on a given Google Cloud Platform project and region. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters', http_method='POST', method_id='gkemulticloud.projects.locations.attachedClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['attachedClusterId', 'validateOnly'], relative_path='v1/{+parent}/attachedClusters', request_field='googleCloudGkemulticloudV1AttachedCluster', request_type_name='GkemulticloudProjectsLocationsAttachedClustersCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific AttachedCluster resource. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters/{attachedClustersId}', http_method='DELETE', method_id='gkemulticloud.projects.locations.attachedClusters.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAttachedClustersDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GenerateAttachedClusterAgentToken(self, request, global_params=None):
        """Generates an access token for a cluster agent.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersGenerateAttachedClusterAgentTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateAttachedClusterAgentToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateAttachedClusterAgentToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters/{attachedClustersId}:generateAttachedClusterAgentToken', http_method='POST', method_id='gkemulticloud.projects.locations.attachedClusters.generateAttachedClusterAgentToken', ordered_params=['attachedCluster'], path_params=['attachedCluster'], query_params=[], relative_path='v1/{+attachedCluster}:generateAttachedClusterAgentToken', request_field='googleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest', request_type_name='GkemulticloudProjectsLocationsAttachedClustersGenerateAttachedClusterAgentTokenRequest', response_type_name='GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Describes a specific AttachedCluster resource.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AttachedCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters/{attachedClustersId}', http_method='GET', method_id='gkemulticloud.projects.locations.attachedClusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAttachedClustersGetRequest', response_type_name='GoogleCloudGkemulticloudV1AttachedCluster', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports creates a new AttachedCluster resource by importing an existing Fleet Membership resource. Attached Clusters created before the introduction of the Anthos Multi-Cloud API can be imported through this method. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters:import', http_method='POST', method_id='gkemulticloud.projects.locations.attachedClusters.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attachedClusters:import', request_field='googleCloudGkemulticloudV1ImportAttachedClusterRequest', request_type_name='GkemulticloudProjectsLocationsAttachedClustersImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AttachedCluster resources on a given Google Cloud project and region.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1ListAttachedClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters', http_method='GET', method_id='gkemulticloud.projects.locations.attachedClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/attachedClusters', request_field='', request_type_name='GkemulticloudProjectsLocationsAttachedClustersListRequest', response_type_name='GoogleCloudGkemulticloudV1ListAttachedClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an AttachedCluster.

      Args:
        request: (GkemulticloudProjectsLocationsAttachedClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/attachedClusters/{attachedClustersId}', http_method='PATCH', method_id='gkemulticloud.projects.locations.attachedClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudGkemulticloudV1AttachedCluster', request_type_name='GkemulticloudProjectsLocationsAttachedClustersPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)