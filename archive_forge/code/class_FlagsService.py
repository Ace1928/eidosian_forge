from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class FlagsService(base_api.BaseApiService):
    """Service class for the flags resource."""
    _NAME = 'flags'

    def __init__(self, client):
        super(SqladminV1beta4.FlagsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all available database flags for Cloud SQL instances.

      Args:
        request: (SqlFlagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FlagsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.flags.list', ordered_params=[], path_params=[], query_params=['databaseVersion'], relative_path='sql/v1beta4/flags', request_field='', request_type_name='SqlFlagsListRequest', response_type_name='FlagsListResponse', supports_download=False)