from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class BillingAccountsLocationsOperationsService(base_api.BaseApiService):
    """Service class for the billingAccounts_locations_operations resource."""
    _NAME = 'billingAccounts_locations_operations'

    def __init__(self, client):
        super(LoggingV2.BillingAccountsLocationsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def ApproveRedaction(self, request, global_params=None):
        """Once the impact assessment completes, the redaction operation will move into WAIT_FOR_USER_APPROVAL stage wherein it's going to wait for the user to approve the redaction operation. Please note that the operation will be in progress at this point and if the user doesn't approve the redaction operation within the grace period, it will be auto-cancelled.The redaction operation can also be approved before operation moves into the WAIT_FOR_USER_APPROVAL stage. In that case redaction process will commence as soon as the impact assessment is complete. This is functionally similar to approving after the operation moves to WAIT_FOR_USER_APPROVAL stage but without any wait time to begin redaction.Once the user approves, the redaction operation will begin redacting the log entries.

      Args:
        request: (LoggingBillingAccountsLocationsOperationsApproveRedactionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApproveRedactionOperationResponse) The response message.
      """
        config = self.GetMethodConfig('ApproveRedaction')
        return self._RunMethod(config, request, global_params=global_params)
    ApproveRedaction.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/operations/{operationsId}:approveRedaction', http_method='GET', method_id='logging.billingAccounts.locations.operations.approveRedaction', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:approveRedaction', request_field='', request_type_name='LoggingBillingAccountsLocationsOperationsApproveRedactionRequest', response_type_name='ApproveRedactionOperationResponse', supports_download=False)

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns google.rpc.Code.UNIMPLEMENTED. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to Code.CANCELLED.

      Args:
        request: (LoggingBillingAccountsLocationsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/operations/{operationsId}:cancel', http_method='POST', method_id='logging.billingAccounts.locations.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:cancel', request_field='cancelOperationRequest', request_type_name='LoggingBillingAccountsLocationsOperationsCancelRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (LoggingBillingAccountsLocationsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/operations/{operationsId}', http_method='GET', method_id='logging.billingAccounts.locations.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingBillingAccountsLocationsOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns UNIMPLEMENTED.

      Args:
        request: (LoggingBillingAccountsLocationsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/locations/{locationsId}/operations', http_method='GET', method_id='logging.billingAccounts.locations.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+name}/operations', request_field='', request_type_name='LoggingBillingAccountsLocationsOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)