from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def Lease(self, request, global_params=None):
    """Leases a dataflow WorkItem to run.

      Args:
        request: (DataflowProjectsLocationsJobsWorkItemsLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LeaseWorkItemResponse) The response message.
      """
    config = self.GetMethodConfig('Lease')
    return self._RunMethod(config, request, global_params=global_params)