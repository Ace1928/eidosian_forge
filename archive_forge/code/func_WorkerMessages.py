from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def WorkerMessages(self, request, global_params=None):
    """Send a worker_message to the service.

      Args:
        request: (DataflowProjectsWorkerMessagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SendWorkerMessagesResponse) The response message.
      """
    config = self.GetMethodConfig('WorkerMessages')
    return self._RunMethod(config, request, global_params=global_params)