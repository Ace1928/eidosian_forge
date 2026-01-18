from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
class EffectiveTagsService(base_api.BaseApiService):
    """Service class for the effectiveTags resource."""
    _NAME = 'effectiveTags'

    def __init__(self, client):
        super(CloudresourcemanagerV3.EffectiveTagsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Return a list of effective tags for the given Google Cloud resource, as specified in `parent`.

      Args:
        request: (CloudresourcemanagerEffectiveTagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEffectiveTagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.effectiveTags.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v3/effectiveTags', request_field='', request_type_name='CloudresourcemanagerEffectiveTagsListRequest', response_type_name='ListEffectiveTagsResponse', supports_download=False)