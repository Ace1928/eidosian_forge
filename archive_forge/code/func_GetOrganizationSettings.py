from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
def GetOrganizationSettings(self, request, global_params=None):
    """Gets the settings for an organization.

      Args:
        request: (SecuritycenterOrganizationsGetOrganizationSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrganizationSettings) The response message.
      """
    config = self.GetMethodConfig('GetOrganizationSettings')
    return self._RunMethod(config, request, global_params=global_params)