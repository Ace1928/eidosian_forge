from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeatureOnlineStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_featureOnlineStores resource."""
    _NAME = 'projects_locations_featureOnlineStores'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeatureOnlineStoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FeatureOnlineStore in a given project and location.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores', http_method='POST', method_id='aiplatform.projects.locations.featureOnlineStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureOnlineStoreId'], relative_path='v1/{+parent}/featureOnlineStores', request_field='googleCloudAiplatformV1FeatureOnlineStore', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single FeatureOnlineStore. The FeatureOnlineStore must not contain any FeatureViews.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}', http_method='DELETE', method_id='aiplatform.projects.locations.featureOnlineStores.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single FeatureOnlineStore.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1FeatureOnlineStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresGetRequest', response_type_name='GoogleCloudAiplatformV1FeatureOnlineStore', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FeatureOnlineStores in a given project and location.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListFeatureOnlineStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/featureOnlineStores', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresListRequest', response_type_name='GoogleCloudAiplatformV1ListFeatureOnlineStoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single FeatureOnlineStore.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}', http_method='PATCH', method_id='aiplatform.projects.locations.featureOnlineStores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1FeatureOnlineStore', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)