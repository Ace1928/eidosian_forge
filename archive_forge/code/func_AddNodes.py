from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
def AddNodes(self, request, global_params=None):
    """Add bare metal nodes to a cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersAddNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('AddNodes')
    return self._RunMethod(config, request, global_params=global_params)