from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class BillingAccountsSinksService(base_api.BaseApiService):
    """Service class for the billingAccounts_sinks resource."""
    _NAME = 'billingAccounts_sinks'

    def __init__(self, client):
        super(LoggingV2.BillingAccountsSinksService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a sink that exports specified log entries to a destination. The export begins upon ingress, unless the sink's writer_identity is not permitted to write to the destination. A sink can export log entries only from the resource owning the sink.

      Args:
        request: (LoggingBillingAccountsSinksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogSink) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks', http_method='POST', method_id='logging.billingAccounts.sinks.create', ordered_params=['parent'], path_params=['parent'], query_params=['customWriterIdentity', 'uniqueWriterIdentity'], relative_path='v2/{+parent}/sinks', request_field='logSink', request_type_name='LoggingBillingAccountsSinksCreateRequest', response_type_name='LogSink', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a sink. If the sink has a unique writer_identity, then that service account is also deleted.

      Args:
        request: (LoggingBillingAccountsSinksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks/{sinksId}', http_method='DELETE', method_id='logging.billingAccounts.sinks.delete', ordered_params=['sinkName'], path_params=['sinkName'], query_params=[], relative_path='v2/{+sinkName}', request_field='', request_type_name='LoggingBillingAccountsSinksDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a sink.

      Args:
        request: (LoggingBillingAccountsSinksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogSink) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks/{sinksId}', http_method='GET', method_id='logging.billingAccounts.sinks.get', ordered_params=['sinkName'], path_params=['sinkName'], query_params=[], relative_path='v2/{+sinkName}', request_field='', request_type_name='LoggingBillingAccountsSinksGetRequest', response_type_name='LogSink', supports_download=False)

    def List(self, request, global_params=None):
        """Lists sinks.

      Args:
        request: (LoggingBillingAccountsSinksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSinksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks', http_method='GET', method_id='logging.billingAccounts.sinks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/sinks', request_field='', request_type_name='LoggingBillingAccountsSinksListRequest', response_type_name='ListSinksResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a sink. This method replaces the values of the destination and filter fields of the existing sink with the corresponding values from the new sink.The updated sink might also have a new writer_identity; see the unique_writer_identity field.

      Args:
        request: (LoggingBillingAccountsSinksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogSink) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks/{sinksId}', http_method='PATCH', method_id='logging.billingAccounts.sinks.patch', ordered_params=['sinkName'], path_params=['sinkName'], query_params=['customWriterIdentity', 'uniqueWriterIdentity', 'updateMask'], relative_path='v2/{+sinkName}', request_field='logSink', request_type_name='LoggingBillingAccountsSinksPatchRequest', response_type_name='LogSink', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a sink. This method replaces the values of the destination and filter fields of the existing sink with the corresponding values from the new sink.The updated sink might also have a new writer_identity; see the unique_writer_identity field.

      Args:
        request: (LoggingBillingAccountsSinksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogSink) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/sinks/{sinksId}', http_method='PUT', method_id='logging.billingAccounts.sinks.update', ordered_params=['sinkName'], path_params=['sinkName'], query_params=['customWriterIdentity', 'uniqueWriterIdentity', 'updateMask'], relative_path='v2/{+sinkName}', request_field='logSink', request_type_name='LoggingBillingAccountsSinksUpdateRequest', response_type_name='LogSink', supports_download=False)