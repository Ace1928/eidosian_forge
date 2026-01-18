from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v2 import iam_v2_messages as messages
class PoliciesOperationsService(base_api.BaseApiService):
    """Service class for the policies_operations resource."""
    _NAME = 'policies_operations'

    def __init__(self, client):
        super(IamV2.PoliciesOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (IamPoliciesOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}/{policiesId2}/operations/{operationsId}', http_method='GET', method_id='iam.policies.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='IamPoliciesOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)