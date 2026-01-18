from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeatureOnlineStoresFeatureViewsService(base_api.BaseApiService):
    """Service class for the projects_locations_featureOnlineStores_featureViews resource."""
    _NAME = 'projects_locations_featureOnlineStores_featureViews'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeatureOnlineStoresFeatureViewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FeatureView in a given FeatureOnlineStore.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews', http_method='POST', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureViewId', 'runSyncImmediately'], relative_path='v1/{+parent}/featureViews', request_field='googleCloudAiplatformV1FeatureView', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}', http_method='DELETE', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def FetchFeatureValues(self, request, global_params=None):
        """Fetch feature values under a FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFetchFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1FetchFeatureValuesResponse) The response message.
      """
        config = self.GetMethodConfig('FetchFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    FetchFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}:fetchFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.fetchFeatureValues', ordered_params=['featureView'], path_params=['featureView'], query_params=[], relative_path='v1/{+featureView}:fetchFeatureValues', request_field='googleCloudAiplatformV1FetchFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFetchFeatureValuesRequest', response_type_name='GoogleCloudAiplatformV1FetchFeatureValuesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1FeatureView) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsGetRequest', response_type_name='GoogleCloudAiplatformV1FeatureView', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FeatureViews in a given FeatureOnlineStore.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListFeatureViewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/featureViews', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsListRequest', response_type_name='GoogleCloudAiplatformV1ListFeatureViewsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}', http_method='PATCH', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1FeatureView', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SearchNearestEntities(self, request, global_params=None):
        """Search the nearest entities under a FeatureView. Search only works for indexable feature view; if a feature view isn't indexable, returns Invalid argument response.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSearchNearestEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchNearestEntitiesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchNearestEntities')
        return self._RunMethod(config, request, global_params=global_params)
    SearchNearestEntities.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}:searchNearestEntities', http_method='POST', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.searchNearestEntities', ordered_params=['featureView'], path_params=['featureView'], query_params=[], relative_path='v1/{+featureView}:searchNearestEntities', request_field='googleCloudAiplatformV1SearchNearestEntitiesRequest', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSearchNearestEntitiesRequest', response_type_name='GoogleCloudAiplatformV1SearchNearestEntitiesResponse', supports_download=False)