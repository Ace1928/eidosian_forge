from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class OrganizationsLocationsInsightTypesInsightsService(base_api.BaseApiService):
    """Service class for the organizations_locations_insightTypes_insights resource."""
    _NAME = 'organizations_locations_insightTypes_insights'

    def __init__(self, client):
        super(RecommenderV1alpha2.OrganizationsLocationsInsightTypesInsightsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the requested insight. Requires the recommender.*.get IAM permission for the specified insight type.

      Args:
        request: (RecommenderOrganizationsLocationsInsightTypesInsightsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Insight) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/organizations/{organizationsId}/locations/{locationsId}/insightTypes/{insightTypesId}/insights/{insightsId}', http_method='GET', method_id='recommender.organizations.locations.insightTypes.insights.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='RecommenderOrganizationsLocationsInsightTypesInsightsGetRequest', response_type_name='GoogleCloudRecommenderV1alpha2Insight', supports_download=False)

    def List(self, request, global_params=None):
        """Lists insights for the specified Cloud Resource. Requires the recommender.*.list IAM permission for the specified insight type.

      Args:
        request: (RecommenderOrganizationsLocationsInsightTypesInsightsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2ListInsightsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/organizations/{organizationsId}/locations/{locationsId}/insightTypes/{insightTypesId}/insights', http_method='GET', method_id='recommender.organizations.locations.insightTypes.insights.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/insights', request_field='', request_type_name='RecommenderOrganizationsLocationsInsightTypesInsightsListRequest', response_type_name='GoogleCloudRecommenderV1alpha2ListInsightsResponse', supports_download=False)

    def MarkAccepted(self, request, global_params=None):
        """Marks the Insight State as Accepted. Users can use this method to indicate to the Recommender API that they have applied some action based on the insight. This stops the insight content from being updated. MarkInsightAccepted can be applied to insights in ACTIVE state. Requires the recommender.*.update IAM permission for the specified insight.

      Args:
        request: (RecommenderOrganizationsLocationsInsightTypesInsightsMarkAcceptedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Insight) The response message.
      """
        config = self.GetMethodConfig('MarkAccepted')
        return self._RunMethod(config, request, global_params=global_params)
    MarkAccepted.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/organizations/{organizationsId}/locations/{locationsId}/insightTypes/{insightTypesId}/insights/{insightsId}:markAccepted', http_method='POST', method_id='recommender.organizations.locations.insightTypes.insights.markAccepted', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markAccepted', request_field='googleCloudRecommenderV1alpha2MarkInsightAcceptedRequest', request_type_name='RecommenderOrganizationsLocationsInsightTypesInsightsMarkAcceptedRequest', response_type_name='GoogleCloudRecommenderV1alpha2Insight', supports_download=False)

    def MarkActive(self, request, global_params=None):
        """Mark the Insight State as Active. Users can use this method to indicate to the Recommender API that a DISMISSED insight has to be marked back as ACTIVE. MarkInsightActive can be applied to insights in DISMISSED state. Requires the recommender.*.update IAM permission for the specified insight type.

      Args:
        request: (RecommenderOrganizationsLocationsInsightTypesInsightsMarkActiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Insight) The response message.
      """
        config = self.GetMethodConfig('MarkActive')
        return self._RunMethod(config, request, global_params=global_params)
    MarkActive.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/organizations/{organizationsId}/locations/{locationsId}/insightTypes/{insightTypesId}/insights/{insightsId}:markActive', http_method='POST', method_id='recommender.organizations.locations.insightTypes.insights.markActive', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markActive', request_field='googleCloudRecommenderV1alpha2MarkInsightActiveRequest', request_type_name='RecommenderOrganizationsLocationsInsightTypesInsightsMarkActiveRequest', response_type_name='GoogleCloudRecommenderV1alpha2Insight', supports_download=False)

    def MarkDismissed(self, request, global_params=None):
        """Mark the Insight State as Dismissed. Users can use this method to indicate to the Recommender API that an ACTIVE insight should be dismissed. MarkInsightDismissed can be applied to insights in ACTIVE state. Requires the recommender.*.update IAM permission for the specified insight type.

      Args:
        request: (RecommenderOrganizationsLocationsInsightTypesInsightsMarkDismissedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Insight) The response message.
      """
        config = self.GetMethodConfig('MarkDismissed')
        return self._RunMethod(config, request, global_params=global_params)
    MarkDismissed.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/organizations/{organizationsId}/locations/{locationsId}/insightTypes/{insightTypesId}/insights/{insightsId}:markDismissed', http_method='POST', method_id='recommender.organizations.locations.insightTypes.insights.markDismissed', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markDismissed', request_field='googleCloudRecommenderV1alpha2MarkInsightDismissedRequest', request_type_name='RecommenderOrganizationsLocationsInsightTypesInsightsMarkDismissedRequest', response_type_name='GoogleCloudRecommenderV1alpha2Insight', supports_download=False)