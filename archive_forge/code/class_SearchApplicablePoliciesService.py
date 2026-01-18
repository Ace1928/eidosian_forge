from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
class SearchApplicablePoliciesService(base_api.BaseApiService):
    """Service class for the searchApplicablePolicies resource."""
    _NAME = 'searchApplicablePolicies'

    def __init__(self, client):
        super(IamV3beta.SearchApplicablePoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Search(self, request, global_params=None):
        """Returns policies (along with the bindings that bind them) that apply to the specified target_query. This means the policies that are bound to the target or any of its ancestors. target_query can be a principal, a principalSet or in the future a resource.

      Args:
        request: (IamSearchApplicablePoliciesSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchApplicablePoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='iam.searchApplicablePolicies.search', ordered_params=[], path_params=[], query_params=['filter', 'pageSize', 'pageToken', 'targetQuery'], relative_path='v3beta/searchApplicablePolicies:search', request_field='', request_type_name='IamSearchApplicablePoliciesSearchRequest', response_type_name='GoogleIamV3betaSearchApplicablePoliciesResponse', supports_download=False)