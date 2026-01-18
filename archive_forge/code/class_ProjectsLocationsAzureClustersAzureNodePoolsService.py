from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAzureClustersAzureNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_azureClusters_azureNodePools resource."""
    _NAME = 'projects_locations_azureClusters_azureNodePools'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAzureClustersAzureNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AzureNodePool, attached to a given AzureCluster. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClusters/{azureClustersId}/azureNodePools', http_method='POST', method_id='gkemulticloud.projects.locations.azureClusters.azureNodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['azureNodePoolId', 'validateOnly'], relative_path='v1/{+parent}/azureNodePools', request_field='googleCloudGkemulticloudV1AzureNodePool', request_type_name='GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific AzureNodePool resource. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClusters/{azureClustersId}/azureNodePools/{azureNodePoolsId}', http_method='DELETE', method_id='gkemulticloud.projects.locations.azureClusters.azureNodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Describes a specific AzureNodePool resource.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureNodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClusters/{azureClustersId}/azureNodePools/{azureNodePoolsId}', http_method='GET', method_id='gkemulticloud.projects.locations.azureClusters.azureNodePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsGetRequest', response_type_name='GoogleCloudGkemulticloudV1AzureNodePool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AzureNodePool resources on a given AzureCluster.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1ListAzureNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClusters/{azureClustersId}/azureNodePools', http_method='GET', method_id='gkemulticloud.projects.locations.azureClusters.azureNodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/azureNodePools', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsListRequest', response_type_name='GoogleCloudGkemulticloudV1ListAzureNodePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an AzureNodePool.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClusters/{azureClustersId}/azureNodePools/{azureNodePoolsId}', http_method='PATCH', method_id='gkemulticloud.projects.locations.azureClusters.azureNodePools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudGkemulticloudV1AzureNodePool', request_type_name='GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)