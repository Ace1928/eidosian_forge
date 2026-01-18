from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1beta import workstations_v1beta_messages as messages
def ListUsable(self, request, global_params=None):
    """Returns all workstation configurations in the specified cluster on which the caller has the "workstations.workstation.create" permission.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUsableWorkstationConfigsResponse) The response message.
      """
    config = self.GetMethodConfig('ListUsable')
    return self._RunMethod(config, request, global_params=global_params)