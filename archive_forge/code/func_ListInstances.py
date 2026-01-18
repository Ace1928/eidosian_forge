from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListInstances(self, request, global_params=None):
    """Lists the instances in the specified instance group and displays information about the named ports. Depending on the specified options, this method can list all instances or only the instances that are running. The orderBy query parameter is not supported.

      Args:
        request: (ComputeRegionInstanceGroupsListInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupsListInstances) The response message.
      """
    config = self.GetMethodConfig('ListInstances')
    return self._RunMethod(config, request, global_params=global_params)