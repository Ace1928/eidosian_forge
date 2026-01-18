from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetXpnResources(self, request, global_params=None):
    """Gets service resources (a.k.a service project) associated with this host project.

      Args:
        request: (ComputeProjectsGetXpnResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectsGetXpnResources) The response message.
      """
    config = self.GetMethodConfig('GetXpnResources')
    return self._RunMethod(config, request, global_params=global_params)