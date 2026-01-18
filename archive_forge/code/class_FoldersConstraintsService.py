from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
class FoldersConstraintsService(base_api.BaseApiService):
    """Service class for the folders_constraints resource."""
    _NAME = 'folders_constraints'

    def __init__(self, client):
        super(OrgpolicyV2.FoldersConstraintsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists constraints that could be applied on the specified resource.

      Args:
        request: (OrgpolicyFoldersConstraintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2ListConstraintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/constraints', http_method='GET', method_id='orgpolicy.folders.constraints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/constraints', request_field='', request_type_name='OrgpolicyFoldersConstraintsListRequest', response_type_name='GoogleCloudOrgpolicyV2ListConstraintsResponse', supports_download=False)