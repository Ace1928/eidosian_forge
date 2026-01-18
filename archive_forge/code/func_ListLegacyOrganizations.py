from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def ListLegacyOrganizations(self, request, global_params=None):
    """Returns a list of legacy organizations.

      Args:
        request: (SasportalCustomersListLegacyOrganizationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListLegacyOrganizationsResponse) The response message.
      """
    config = self.GetMethodConfig('ListLegacyOrganizations')
    return self._RunMethod(config, request, global_params=global_params)