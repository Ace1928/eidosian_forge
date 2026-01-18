from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListNodes(self, request, global_params=None):
    """Lists nodes in the node group.

      Args:
        request: (ComputeNodeGroupsListNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroupsListNodes) The response message.
      """
    config = self.GetMethodConfig('ListNodes')
    return self._RunMethod(config, request, global_params=global_params)