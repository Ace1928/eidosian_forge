from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalOrganizationOperationsService(base_api.BaseApiService):
    """Service class for the globalOrganizationOperations resource."""
    _NAME = 'globalOrganizationOperations'

    def __init__(self, client):
        super(ComputeBeta.GlobalOrganizationOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified Operations resource.

      Args:
        request: (ComputeGlobalOrganizationOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeGlobalOrganizationOperationsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalOrganizationOperations.delete', ordered_params=['operation'], path_params=['operation'], query_params=['parentId'], relative_path='locations/global/operations/{operation}', request_field='', request_type_name='ComputeGlobalOrganizationOperationsDeleteRequest', response_type_name='ComputeGlobalOrganizationOperationsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified Operations resource. Gets a list of operations by making a `list()` request.

      Args:
        request: (ComputeGlobalOrganizationOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalOrganizationOperations.get', ordered_params=['operation'], path_params=['operation'], query_params=['parentId'], relative_path='locations/global/operations/{operation}', request_field='', request_type_name='ComputeGlobalOrganizationOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of Operation resources contained within the specified organization.

      Args:
        request: (ComputeGlobalOrganizationOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OperationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalOrganizationOperations.list', ordered_params=[], path_params=[], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'parentId', 'returnPartialSuccess'], relative_path='locations/global/operations', request_field='', request_type_name='ComputeGlobalOrganizationOperationsListRequest', response_type_name='OperationList', supports_download=False)