from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def ExecuteBatchDml(self, request, global_params=None):
    """Executes a batch of SQL DML statements. This method allows many statements to be run with lower latency than submitting them sequentially with ExecuteSql. Statements are executed in sequential order. A request can succeed even if a statement fails. The ExecuteBatchDmlResponse.status field in the response provides information about the statement that failed. Clients must inspect this field to determine whether an error occurred. Execution stops after the first failed statement; the remaining statements are not executed.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteBatchDmlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteBatchDmlResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteBatchDml')
    return self._RunMethod(config, request, global_params=global_params)