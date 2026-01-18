from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListAssociations(self, request, global_params=None):
    """Lists associations of a specified target, i.e., organization or folder.

      Args:
        request: (ComputeOrganizationSecurityPoliciesListAssociationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrganizationSecurityPoliciesListAssociationsResponse) The response message.
      """
    config = self.GetMethodConfig('ListAssociations')
    return self._RunMethod(config, request, global_params=global_params)