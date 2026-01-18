from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class InsightTypesService(base_api.BaseApiService):
    """Service class for the insightTypes resource."""
    _NAME = 'insightTypes'

    def __init__(self, client):
        super(RecommenderV1alpha2.InsightTypesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists available InsightTypes. No IAM permissions are required.

      Args:
        request: (RecommenderInsightTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2ListInsightTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='recommender.insightTypes.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/insightTypes', request_field='', request_type_name='RecommenderInsightTypesListRequest', response_type_name='GoogleCloudRecommenderV1alpha2ListInsightTypesResponse', supports_download=False)