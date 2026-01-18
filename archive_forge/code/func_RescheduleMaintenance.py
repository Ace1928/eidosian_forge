from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.memcache.v1 import memcache_v1_messages as messages
def RescheduleMaintenance(self, request, global_params=None):
    """Reschedules upcoming maintenance event.

      Args:
        request: (MemcacheProjectsLocationsInstancesRescheduleMaintenanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RescheduleMaintenance')
    return self._RunMethod(config, request, global_params=global_params)