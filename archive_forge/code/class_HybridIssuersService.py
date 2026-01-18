from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class HybridIssuersService(base_api.BaseApiService):
    """Service class for the hybrid_issuers resource."""
    _NAME = 'hybrid_issuers'

    def __init__(self, client):
        super(ApigeeV1.HybridIssuersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists hybrid services and its trusted issuers service account ids. This api is authenticated and unauthorized(allow all the users) and used by runtime authn-authz service to query control plane's issuer service account ids.

      Args:
        request: (ApigeeHybridIssuersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListHybridIssuersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/hybrid/issuers', http_method='GET', method_id='apigee.hybrid.issuers.list', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeHybridIssuersListRequest', response_type_name='GoogleCloudApigeeV1ListHybridIssuersResponse', supports_download=False)