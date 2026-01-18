from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.kmsinventory.v1 import kmsinventory_v1_messages as messages
class ProjectsCryptoKeysService(base_api.BaseApiService):
    """Service class for the projects_cryptoKeys resource."""
    _NAME = 'projects_cryptoKeys'

    def __init__(self, client):
        super(KmsinventoryV1.ProjectsCryptoKeysService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns cryptographic keys managed by Cloud KMS in a given Cloud project. Note that this data is sourced from snapshots, meaning it may not completely reflect the actual state of key metadata at call time.

      Args:
        request: (KmsinventoryProjectsCryptoKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCryptoKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/cryptoKeys', http_method='GET', method_id='kmsinventory.projects.cryptoKeys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/cryptoKeys', request_field='', request_type_name='KmsinventoryProjectsCryptoKeysListRequest', response_type_name='ListCryptoKeysResponse', supports_download=False)