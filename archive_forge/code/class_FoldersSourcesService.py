from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class FoldersSourcesService(base_api.BaseApiService):
    """Service class for the folders_sources resource."""
    _NAME = 'folders_sources'

    def __init__(self, client):
        super(SecuritycenterV2.FoldersSourcesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all sources belonging to an organization.

      Args:
        request: (SecuritycenterFoldersSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/sources', http_method='GET', method_id='securitycenter.folders.sources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/sources', request_field='', request_type_name='SecuritycenterFoldersSourcesListRequest', response_type_name='ListSourcesResponse', supports_download=False)