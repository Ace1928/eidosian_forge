from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class BillingAccountsLogsService(base_api.BaseApiService):
    """Service class for the billingAccounts_logs resource."""
    _NAME = 'billingAccounts_logs'

    def __init__(self, client):
        super(LoggingV2.BillingAccountsLogsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes all the log entries in a log for the _Default Log Bucket. The log reappears if it receives new entries. Log entries written shortly before the delete operation might not be deleted. Entries received after the delete operation with a timestamp before the operation will be deleted.

      Args:
        request: (LoggingBillingAccountsLogsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/logs/{logsId}', http_method='DELETE', method_id='logging.billingAccounts.logs.delete', ordered_params=['logName'], path_params=['logName'], query_params=[], relative_path='v2/{+logName}', request_field='', request_type_name='LoggingBillingAccountsLogsDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the logs in projects, organizations, folders, or billing accounts. Only logs that have entries are listed.

      Args:
        request: (LoggingBillingAccountsLogsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLogsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/logs', http_method='GET', method_id='logging.billingAccounts.logs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'resourceNames'], relative_path='v2/{+parent}/logs', request_field='', request_type_name='LoggingBillingAccountsLogsListRequest', response_type_name='ListLogsResponse', supports_download=False)