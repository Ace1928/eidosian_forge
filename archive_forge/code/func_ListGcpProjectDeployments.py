from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def ListGcpProjectDeployments(self, request, global_params=None):
    """Returns a list of SAS deployments associated with current GCP project. Includes whether SAS analytics has been enabled or not.

      Args:
        request: (SasportalCustomersListGcpProjectDeploymentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListGcpProjectDeploymentsResponse) The response message.
      """
    config = self.GetMethodConfig('ListGcpProjectDeployments')
    return self._RunMethod(config, request, global_params=global_params)