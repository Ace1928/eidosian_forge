from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p1beta1 import cloudasset_v1p1beta1_messages as messages
class ResourcesService(base_api.BaseApiService):
    """Service class for the resources resource."""
    _NAME = 'resources'

    def __init__(self, client):
        super(CloudassetV1p1beta1.ResourcesService, self).__init__(client)
        self._upload_configs = {}

    def SearchAll(self, request, global_params=None):
        """Searches all the resources within a given accessible CRM scope (project/folder/organization). This RPC gives callers especially administrators the ability to search all the resources within a scope, even if they don't have `.get` permission of all the resources. Callers should have `cloud.assets.SearchAllResources` permission on the requested scope, otherwise the request will be rejected.

      Args:
        request: (CloudassetResourcesSearchAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchAllResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchAll')
        return self._RunMethod(config, request, global_params=global_params)
    SearchAll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1p1beta1/{v1p1beta1Id}/{v1p1beta1Id1}/resources:searchAll', http_method='GET', method_id='cloudasset.resources.searchAll', ordered_params=['scope'], path_params=['scope'], query_params=['assetTypes', 'orderBy', 'pageSize', 'pageToken', 'query'], relative_path='v1p1beta1/{+scope}/resources:searchAll', request_field='', request_type_name='CloudassetResourcesSearchAllRequest', response_type_name='SearchAllResourcesResponse', supports_download=False)