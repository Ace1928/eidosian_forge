from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2beta2 import cloudtasks_v2beta2_messages as messages
def RenewLease(self, request, global_params=None):
    """Renew the current lease of a pull task. The worker can use this method to extend the lease by a new duration, starting from now. The new task lease will be returned in the task's schedule_time.

      Args:
        request: (CloudtasksProjectsLocationsQueuesTasksRenewLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Task) The response message.
      """
    config = self.GetMethodConfig('RenewLease')
    return self._RunMethod(config, request, global_params=global_params)