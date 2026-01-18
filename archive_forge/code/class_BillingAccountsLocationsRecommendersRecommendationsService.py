from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class BillingAccountsLocationsRecommendersRecommendationsService(base_api.BaseApiService):
    """Service class for the billingAccounts_locations_recommenders_recommendations resource."""
    _NAME = 'billingAccounts_locations_recommenders_recommendations'

    def __init__(self, client):
        super(RecommenderV1alpha2.BillingAccountsLocationsRecommendersRecommendationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the requested recommendation. Requires the recommender.*.get IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}', http_method='GET', method_id='recommender.billingAccounts.locations.recommenders.recommendations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsGetRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists recommendations for the specified Cloud Resource. Requires the recommender.*.list IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2ListRecommendationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations', http_method='GET', method_id='recommender.billingAccounts.locations.recommenders.recommendations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/recommendations', request_field='', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsListRequest', response_type_name='GoogleCloudRecommenderV1alpha2ListRecommendationsResponse', supports_download=False)

    def MarkActive(self, request, global_params=None):
        """Mark the Recommendation State as Active. Users can use this method to indicate to the Recommender API that a DISMISSED recommendation has to be marked back as ACTIVE. MarkRecommendationActive can be applied to recommendations in DISMISSED state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkActiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('MarkActive')
        return self._RunMethod(config, request, global_params=global_params)
    MarkActive.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}:markActive', http_method='POST', method_id='recommender.billingAccounts.locations.recommenders.recommendations.markActive', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markActive', request_field='googleCloudRecommenderV1alpha2MarkRecommendationActiveRequest', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkActiveRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)

    def MarkClaimed(self, request, global_params=None):
        """Marks the Recommendation State as Claimed. Users can use this method to indicate to the Recommender API that they are starting to apply the recommendation themselves. This stops the recommendation content from being updated. Associated insights are frozen and placed in the ACCEPTED state. MarkRecommendationClaimed can be applied to recommendations in CLAIMED or ACTIVE state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkClaimedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('MarkClaimed')
        return self._RunMethod(config, request, global_params=global_params)
    MarkClaimed.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}:markClaimed', http_method='POST', method_id='recommender.billingAccounts.locations.recommenders.recommendations.markClaimed', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markClaimed', request_field='googleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkClaimedRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)

    def MarkDismissed(self, request, global_params=None):
        """Mark the Recommendation State as Dismissed. Users can use this method to indicate to the Recommender API that an ACTIVE recommendation has to be marked back as DISMISSED. MarkRecommendationDismissed can be applied to recommendations in ACTIVE state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkDismissedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('MarkDismissed')
        return self._RunMethod(config, request, global_params=global_params)
    MarkDismissed.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}:markDismissed', http_method='POST', method_id='recommender.billingAccounts.locations.recommenders.recommendations.markDismissed', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markDismissed', request_field='googleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkDismissedRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)

    def MarkFailed(self, request, global_params=None):
        """Marks the Recommendation State as Failed. Users can use this method to indicate to the Recommender API that they have applied the recommendation themselves, and the operation failed. This stops the recommendation content from being updated. Associated insights are frozen and placed in the ACCEPTED state. MarkRecommendationFailed can be applied to recommendations in ACTIVE, CLAIMED, SUCCEEDED, or FAILED state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkFailedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('MarkFailed')
        return self._RunMethod(config, request, global_params=global_params)
    MarkFailed.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}:markFailed', http_method='POST', method_id='recommender.billingAccounts.locations.recommenders.recommendations.markFailed', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markFailed', request_field='googleCloudRecommenderV1alpha2MarkRecommendationFailedRequest', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkFailedRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)

    def MarkSucceeded(self, request, global_params=None):
        """Marks the Recommendation State as Succeeded. Users can use this method to indicate to the Recommender API that they have applied the recommendation themselves, and the operation was successful. This stops the recommendation content from being updated. Associated insights are frozen and placed in the ACCEPTED state. MarkRecommendationSucceeded can be applied to recommendations in ACTIVE, CLAIMED, SUCCEEDED, or FAILED state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkSucceededRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
        config = self.GetMethodConfig('MarkSucceeded')
        return self._RunMethod(config, request, global_params=global_params)
    MarkSucceeded.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations/{locationsId}/recommenders/{recommendersId}/recommendations/{recommendationsId}:markSucceeded', http_method='POST', method_id='recommender.billingAccounts.locations.recommenders.recommendations.markSucceeded', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:markSucceeded', request_field='googleCloudRecommenderV1alpha2MarkRecommendationSucceededRequest', request_type_name='RecommenderBillingAccountsLocationsRecommendersRecommendationsMarkSucceededRequest', response_type_name='GoogleCloudRecommenderV1alpha2Recommendation', supports_download=False)