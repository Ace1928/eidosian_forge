from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def GetExecutionDetails(self, request, global_params=None):
    """Request detailed information about the execution status of the job. EXPERIMENTAL. This API is subject to change or removal without notice.

      Args:
        request: (DataflowProjectsLocationsJobsGetExecutionDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobExecutionDetails) The response message.
      """
    config = self.GetMethodConfig('GetExecutionDetails')
    return self._RunMethod(config, request, global_params=global_params)