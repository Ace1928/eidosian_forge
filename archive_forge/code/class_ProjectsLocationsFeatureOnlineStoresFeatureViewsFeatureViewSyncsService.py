from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsService(base_api.BaseApiService):
    """Service class for the projects_locations_featureOnlineStores_featureViews_featureViewSyncs resource."""
    _NAME = 'projects_locations_featureOnlineStores_featureViews_featureViewSyncs'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single FeatureViewSync.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1FeatureViewSync) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}/featureViewSyncs/{featureViewSyncsId}', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.featureViewSyncs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsGetRequest', response_type_name='GoogleCloudAiplatformV1beta1FeatureViewSync', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FeatureViewSyncs in a given FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListFeatureViewSyncsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/featureOnlineStores/{featureOnlineStoresId}/featureViews/{featureViewsId}/featureViewSyncs', http_method='GET', method_id='aiplatform.projects.locations.featureOnlineStores.featureViews.featureViewSyncs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/featureViewSyncs', request_field='', request_type_name='AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListFeatureViewSyncsResponse', supports_download=False)