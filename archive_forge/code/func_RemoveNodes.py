from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
def RemoveNodes(self, request, global_params=None):
    """Remove bare metal nodes from a cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersRemoveNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('RemoveNodes')
    return self._RunMethod(config, request, global_params=global_params)