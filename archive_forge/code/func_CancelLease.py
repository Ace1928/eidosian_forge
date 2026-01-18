from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2beta2 import cloudtasks_v2beta2_messages as messages
def CancelLease(self, request, global_params=None):
    """Cancel a pull task's lease. The worker can use this method to cancel a task's lease by setting its schedule_time to now. This will make the task available to be leased to the next caller of LeaseTasks.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksCancelLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Task) The response message.
      """
    config = self.GetMethodConfig('CancelLease')
    return self._RunMethod(config, request, global_params=global_params)