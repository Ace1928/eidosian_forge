from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def Aggregated(self, request, global_params=None):
    """List the jobs of a project across all regions. **Note:** This method doesn't support filtering the list of jobs by name.

      Args:
        request: (DataflowProjectsJobsAggregatedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobsResponse) The response message.
      """
    config = self.GetMethodConfig('Aggregated')
    return self._RunMethod(config, request, global_params=global_params)